import logging
import queue
import threading
import time

import cv2

logger = logging.getLogger(__name__)


class VideoStreamer:
    """Background HLS ingest. Hands out frames + arrival timestamps only.

    Knows nothing about detection. A blocking bounded queue is used on
    purpose: decoded frames are never dropped. If the consumer stalls,
    backpressure raises latency instead of skipping frames.
    """

    def __init__(self, url, max_queue=3, backoff_min=0.5, backoff_max=10.0,
                 reconnect=True, max_retries=None, gap_thresh_sec=0.5):
        self.url = url
        self._queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread = None
        self._backoff_min = backoff_min
        self._backoff_max = backoff_max
        self._reconnect = bool(reconnect)
        self._max_retries = max_retries
        self._gap_thresh = float(gap_thresh_sec)
        self._lock = threading.Lock()
        self._last_gap = None
        self.frames_delivered = 0
        self.connects = 0
        self.reconnects = 0
        # Source frame rate from CAP_PROP_FPS; 0.0 until the stream is open.
        # Consumers pace playback against effective_fps() because FFmpeg's HLS
        # demuxer buffers whole segments and hands them over in one burst
        # (~0ms gaps, then a multi-second stall), so an unpaced consumer would
        # race through each burst -> fast-forward.
        self.native_fps = 0.0

    def effective_fps(self, default=25.0):
        """Source frame rate, or `default` when FFmpeg reports 0/NaN.

        HLS is unreliable here: CAP_PROP_FPS often comes back 0, NaN, or a
        raw timebase such as 90000. `_run` already clamps those to 25.0, so
        this only guards the window before the first successful open.
        """
        fps = self.native_fps
        if not fps or fps != fps or fps <= 0 or fps > 120:
            return default
        return fps

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, join_timeout=2.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(join_timeout)
            self._thread = None

    def get(self, timeout=1.0):
        """Return (frame, arrival_mono, arrival_wall) or None when no new frame."""
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        frame, arrival, wall, gap = item
        with self._lock:
            self._last_gap = gap
            self.frames_delivered += 1
        return frame, arrival, wall

    def read(self, timeout=1.0):
        """Frame-only view for simple consumers (harvester). Blocking until a
        frame is ready or timeout; returns None on timeout."""
        item = self.get(timeout=timeout)
        return item[0] if item else None

    def last_gap(self):
        with self._lock:
            return self._last_gap

    def is_frozen(self):
        """Point-in-time jitter guard: True only for frames delivered with a
        decode-gap > gap_thresh (a late arrival right after an HLS stall, or
        the re-IDs it triggers). Burst members after the late frame return to
        normal: the per-zone permanent latch + spatial-temporal dedup block
        their duplicate ids, so counting no longer has to freeze a time
        window after every stall."""
        with self._lock:
            return self._last_gap is not None and self._last_gap > self._gap_thresh

    def _run(self):
        backoff = self._backoff_min
        attempts = 0
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                cap.release()
                attempts += 1
                if not self._reconnect:
                    return
                if self._max_retries is not None and attempts > self._max_retries:
                    logger.error("gave up on %s after %d failed connects",
                                 self.url, attempts)
                    return
                logger.warning("could not open %s (attempt %d), retry in %.1fs",
                               self.url, attempts, backoff)
                if self._stop.wait(backoff):
                    return
                backoff = min(backoff * 2, self._backoff_max)
                continue
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Read the source rate once per connect. HLS reports 0/NaN or a raw
            # timebase (e.g. 90000) often enough that a sane range is the only
            # safe option; 25.0 is the standard Indonesian CCTV rate.
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            if not native_fps or native_fps != native_fps or native_fps <= 0 \
                    or native_fps > 120:
                native_fps = 25.0
            with self._lock:
                self.native_fps = float(native_fps)
            self.connects += 1
            attempts = 0
            if self.connects > 1:
                self.reconnects += 1
                logger.info("reconnected to %s", self.url)
            backoff = self._backoff_min
            logger.info("%s opened at %.1f fps", self.url, native_fps)
            try:
                self._pump(cap)
            finally:
                cap.release()
            if not self._stop.is_set():
                if not self._reconnect:
                    return
                logger.warning("stream read failed, reconnecting")
                self._stop.wait(backoff)

    def _pump(self, cap):
        prev = None
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                return
            arrival = time.monotonic()
            wall = time.time()
            # Gap measured by the reader: backpressure from a slow consumer
            # must NOT mark frames as jitter — only a decode stall does.
            gap = None if prev is None else arrival - prev
            prev = arrival
            # Blocking put = no frame is ever skipped here.
            # ponytail: if the consumer is permanently slower than the stream,
            # upstream buffering grows without bound. Add a qsize watchdog
            # only if measured latency actually matters.
            while not self._stop.is_set():
                try:
                    self._queue.put((frame, arrival, wall, gap), timeout=0.25)
                    break
                except queue.Full:
                    continue
            if self._stop.is_set():
                return
