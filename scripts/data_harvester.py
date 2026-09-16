"""Periodically save frames from a VideoStreamer for later annotation."""
from __future__ import annotations

import argparse
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from src.utils.logger import get_logger
from src.stream.video_streamer import VideoStreamer

log = get_logger(__name__)


class DataHarvester:
    """Drops one frame every `interval_sec` to `output_dir` as JPEG."""

    def __init__(
        self,
        streamer: VideoStreamer,
        output_dir: str,
        *,
        interval_sec: float = 2.0,
        prefix: str = "frame",
        jpeg_quality: int = 95,
    ) -> None:
        if interval_sec <= 0:
            raise ValueError("interval_sec must be > 0")
        self._streamer = streamer
        self._output_dir = Path(output_dir)
        self._interval = interval_sec
        self._prefix = prefix
        self._quality = int(jpeg_quality)
        self._stop = False
        self._frame_idx = 0
        self._saved = 0

    def request_stop(self, *_: object) -> None:
        log.info("Stop requested")
        self._stop = True

    def run(self, max_frames: Optional[int] = None) -> int:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

        self._streamer.start()
        log.info("Harvesting to %s every %.2fs", self._output_dir, self._interval)

        last_save = 0.0
        try:
            while not self._stop:
                frame = self._streamer.read()
                if frame is None:
                    continue
                self._frame_idx += 1

                now = time.monotonic()
                if now - last_save < self._interval:
                    if max_frames is not None and self._frame_idx >= max_frames:
                        break
                    continue

                if self._save(frame):
                    last_save = now
                    self._saved += 1
                    if self._saved % 20 == 0:
                        log.info("Saved %d frames", self._saved)

                if max_frames is not None and self._frame_idx >= max_frames:
                    break
        finally:
            self._streamer.stop()
            log.info("Done. Total saved: %d", self._saved)
        return self._saved

    def _save(self, frame) -> bool:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S_%f")[:-3]
        name = f"{self._prefix}_{ts}_{self._frame_idx:08d}.jpg"
        path = self._output_dir / name
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
        if not ok:
            log.warning("imencode failed for frame %d", self._frame_idx)
            return False
        try:
            path.write_bytes(buf.tobytes())
        except OSError as exc:
            log.warning("Write failed (%s): %s", path, exc)
            return False
        return True


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ATCS dataset harvester")
    p.add_argument("--source", required=True, help="HLS URL, RTSP URL, or USB index")
    p.add_argument("--out", default="dataset/raw_images", help="Output directory")
    p.add_argument("--interval", type=float, default=2.0, help="Seconds between saves")
    p.add_argument("--prefix", default="frame")
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--reconnect/--no-reconnect", dest="reconnect", default=True)
    p.add_argument("--max-retries", type=int, default=5)
    return p.parse_args()


def main() -> None:
    from src.utils.logger import setup_logging

    setup_logging()
    args = _parse_args()
    streamer = VideoStreamer(
        args.source,
        reconnect=args.reconnect,
        max_retries=args.max_retries,
    )
    harvester = DataHarvester(
        streamer,
        args.out,
        interval_sec=args.interval,
        prefix=args.prefix,
        jpeg_quality=args.quality,
    )
    harvester.run(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
