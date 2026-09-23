"""ATCS orchestrator: settings.yaml -> stream thread -> per-frame
Grab -> Detect -> Count -> Log -> Render -> Display -> Audit video.

Asynchronous pipeline, three threads per node:
  ingest  VideoStreamer decodes HLS/RTSP into a bounded queue (native rate)
  display consumes every frame, draws the newest overlay, shows / publishes it
  ai      samples the newest raw frame at target_fps (focused) or throttle_fps
          (background), runs RT-DETR + ByteTrack, stores the overlay

The display loop never waits on the model, so the video stays fluid at the
source's native rate while the boxes refresh at the AI rate. The display is
the only consumer of the stream queue: the AI samples the shared newest frame
instead, otherwise the two would split frames ~50/50 and neither would hit
its target rate."""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import uvicorn
import yaml

from src.api.server import app, get_active_focus, set_latest_frame, update_node_data
from src.core.detector import Detector, centroids
from src.core.zone_counter import ZoneCounter
from src.stream.video_streamer import VideoStreamer
from src.ui.renderer import Renderer, ZoneEditor
from src.utils.logger import AuditLogger, get_logger, setup_logging

log = get_logger(__name__)


def build_zones(zone_cfgs):
    zones = []
    for zc in zone_cfgs:
        zones.append(sv.PolygonZone(polygon=np.array(zc["polygon"], dtype=np.float32)))
    return zones


def save_zones(config_path, zones, zone_cfgs):
    with open(config_path, "r", encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c["zones"] = [
        {"name": zc.get("name", "zone"),
         "polygon": [[int(p[0]), int(p[1])] for p in z.polygon]}
        for z, zc in zip(zones, zone_cfgs)
    ]
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(c, f, allow_unicode=True, sort_keys=False)


def open_audit_writer(frame, fps):
    """Timestamped annotated-video sink in logs/. mp4v first, AVI/XVID
    fallback for Windows builds without an MP4 backend."""
    h, w = frame.shape[:2]
    ts = time.strftime("%Y%m%d_%H%M%S")
    Path("logs").mkdir(exist_ok=True)
    for fourcc, ext in (("mp4v", "mp4"), ("XVID", "avi")):
        path = f"logs/audit_video_{ts}.{ext}"
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if vw.isOpened():
            return vw, path
        vw.release()
        Path(path).unlink(missing_ok=True)
    log.warning("no working VideoWriter codec; audit video disabled")
    return None, None


class FrameBus:
    """Newest raw frame (display -> AI) and newest overlay (AI -> display).

    The display thread hands the AI a reference to the pristine frame and
    draws on a copy, so a detect() in flight never sees a half-drawn image.
    Both slots sit behind one Condition: the AI blocks on `wait_raw` until a
    frame newer than the one it last used shows up, which is what lets a
    throttled node skip the backlog and always infer on the freshest frame.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._raw = None
        self._seq = 0
        self._overlay = None

    def publish_raw(self, frame, arrival, wall):
        with self._cond:
            self._raw = (frame, arrival, wall)
            self._seq += 1
            self._cond.notify_all()

    def wait_raw(self, last_seq, stop_event, timeout=0.0):
        """Return ((frame, arrival, wall), seq) for a frame newer than
        last_seq, or (None, last_seq) on timeout / stop."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._seq == last_seq:
                if stop_event.is_set():
                    return None, last_seq
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, last_seq
                self._cond.wait(remaining)
            return self._raw, self._seq

    def set_overlay(self, overlay):
        with self._cond:
            self._overlay = overlay

    def get_overlay(self):
        with self._cond:
            return self._overlay


class StageProfiler:
    """Per-stage latency, split across the AI and display loops.

    Diagnostics only. AI stages are bucketed by object count so the Low/High
    groups show whether inference latency tracks traffic volume; display
    stages show the per-frame cost of drawing + publishing. A lock guards the
    counters because the two worker threads tick it concurrently.
    """

    def __init__(self, label, window=5.0):
        self.label = label
        self.window = window
        self._lock = threading.Lock()
        self._reset()

    def _reset(self):
        self.t0 = time.perf_counter()
        self.ai_frames = 0
        self.disp_frames = 0
        self.objs = 0
        self.ai_totals = {}
        self.disp_totals = {}
        self.low = [0, 0, 0.0]    # frames, object_sum, infer_sum  (objs < 10)
        self.high = [0, 0, 0.0]   # frames, object_sum, infer_sum  (objs > 15)

    def tick_ai(self, objs, **stages):
        with self._lock:
            self.ai_frames += 1
            self.objs += objs
            for name, secs in stages.items():
                self.ai_totals[name] = self.ai_totals.get(name, 0.0) + secs
            bucket = self.low if objs < 10 else self.high if objs > 15 else None
            if bucket is not None:
                bucket[0] += 1
                bucket[1] += objs
                bucket[2] += stages.get("infer", 0.0)

    def tick_display(self, **stages):
        with self._lock:
            self.disp_frames += 1
            for name, secs in stages.items():
                self.disp_totals[name] = self.disp_totals.get(name, 0.0) + secs

    def maybe_log(self):
        lines = []
        with self._lock:
            elapsed = time.perf_counter() - self.t0
            if self.ai_frames == 0 or elapsed < self.window:
                return
            ai = " ".join("%s=%.1fms" % (k, 1000.0 * v / self.ai_frames)
                          for k, v in self.ai_totals.items())
            disp = (" ".join("%s=%.1fms" % (k, 1000.0 * v / self.disp_frames)
                             for k, v in self.disp_totals.items())
                    if self.disp_frames else "n/a")
            lines.append("[prof %s] ai %.1f fps objs/frame=%.1f | %s"
                         % (self.label, self.ai_frames / elapsed,
                            self.objs / self.ai_frames, ai))
            lines.append("[prof %s] disp %.1f fps | %s"
                         % (self.label, self.disp_frames / elapsed, disp))
            for name, (n, osum, isum) in (("low<10", self.low), ("high>15", self.high)):
                if n:
                    lines.append("[prof %s]   %s: %d frames, objs/frame=%.1f, infer=%.1fms"
                                 % (self.label, name, n, osum / n, 1000.0 * isum / n))
            self._reset()
        for line in lines:
            log.info("%s", line)


def _ai_worker(node_id, bus, streamer, det, counter, gpu_lock, frame_dt, idle_dt,
               stop_event, prof, reset_flag, t_start, audit=None, debug=False,
               publish_api=False, force_focus=False):
    """Throttled inference loop. Samples the newest raw frame, runs the model,
    and stores the overlay for the display thread. Paced to frame_dt when this
    node is focused, idle_dt otherwise (Lazy Inference).

    `force_focus` pins a node to frame_dt regardless of the API focus state.
    Single-node mode sets it: there is only one stream and no map to switch
    away to, so it must always run at target_fps."""
    last_seq = 0
    next_ai = time.perf_counter()
    while not stop_event.is_set():
        now = time.perf_counter()
        if next_ai > now:
            stop_event.wait(next_ai - now)
            continue

        focused = force_focus or get_active_focus() == node_id
        dt = frame_dt if focused else idle_dt

        # Non-blocking: take whatever is newest right now. If the display has
        # not produced a fresh frame yet, wait one interval and retry.
        t = time.perf_counter()
        raw, last_seq = bus.wait_raw(last_seq, stop_event, timeout=0.0)
        t_acquire = time.perf_counter() - t
        if raw is None:
            next_ai = time.perf_counter() + dt
            continue
        next_ai = max(next_ai + dt, time.perf_counter())
        frame, arrival, wall = raw

        if reset_flag["v"]:
            counter.reset()
            reset_flag["v"] = False

        try:
            # ---- [RT-DETR + ByteTrack INFERENCE] ----
            # detect() runs on the shared engine; gpu_lock serializes calls so
            # one TensorRT context serves all nodes. t_infer includes the wait
            # for that lock, which is the GPU-contention signal.
            t = time.perf_counter()
            with gpu_lock:
                raw_dets = det.detect(frame)
            t_infer = time.perf_counter() - t
            t = time.perf_counter()
            dets = counter.track(raw_dets, frame)
            t_track = time.perf_counter() - t
            events = counter.update(dets, now=arrival,
                                    freeze=streamer.is_frozen())
        except Exception as e:
            log.warning("node %s detector: %s", node_id, e)
            continue

        if audit:
            for ev in events:
                audit.write(ev)
        if debug and events:
            for ev in events:
                log.info("COUNT %s id=%d conf=%.2f zone=%d (total=%d)",
                         ev["class_name"], ev["track_id"], ev["confidence"],
                         ev["zone"], counter.total)

        snap = None
        if len(dets) and dets.tracker_id is not None:
            snap = (dets.xyxy, dets.tracker_id, dets.class_id,
                    centroids(dets.xyxy).astype(int), dets.confidence)
        # Copy the live containers so the display thread reads a stable view.
        bus.set_overlay({
            "snap": snap,
            "counts": dict(counter.total_by_class),
            "trails": {k: list(v) for k, v in counter.trails.items()},
        })

        if publish_api:
            by_class = defaultdict(int)
            if len(dets) and dets.class_id is not None:
                for cid in dets.class_id:
                    by_class[int(cid)] += 1
            update_node_data(node_id, {
                "motor": by_class.get(3, 0),
                "mobil": by_class.get(2, 0),
                "truk": by_class.get(5, 0) + by_class.get(7, 0),
                "occupancy": len(dets),
                "fps": counter.n_frames / max(time.monotonic() - t_start, 1e-6),
            })

        prof.tick_ai(len(dets), infer=t_infer, track=t_track, acquire=t_acquire)
        prof.maybe_log()


def _display_worker(streamer, bus, renderer, stop_event, prof, sink, deadline=None):
    """Fast loop. Sole consumer of the stream queue: pulls every frame at the
    source's native rate, overlays the newest AI result, and hands the frame
    to `sink` (imshow or MJPEG publish). Never blocks on inference.

    Pacing matters because FFmpeg's HLS demuxer buffers whole segments and
    releases them in a burst (~0ms gaps, then a multi-second stall). Draining
    that burst unpaced looks like fast-forward, so each frame is spaced to the
    source rate and the bounded queue backpressures the pump into real time."""
    next_frame = time.monotonic()
    while not stop_event.is_set():
        if deadline is not None and time.monotonic() > deadline:
            stop_event.set()
            break
        item = streamer.get(timeout=1.0)
        if item is None:
            continue
        frame, arrival, wall = item

        t = time.perf_counter()
        bus.publish_raw(frame, arrival, wall)
        t_acquire = time.perf_counter() - t

        overlay = bus.get_overlay()
        # Draw on a copy: the array handed to the AI stays pristine.
        out = frame.copy()
        t = time.perf_counter()
        if renderer is not None and overlay is not None:
            renderer.draw(out, overlay["snap"], overlay["counts"],
                          trails=overlay["trails"])
        t_render = time.perf_counter() - t

        t = time.perf_counter()
        sink(out)
        t_publish = time.perf_counter() - t

        prof.tick_display(acquire=t_acquire, render=t_render, publish=t_publish)

        # Meter the frame to the source rate, before pulling the next one.
        # If playback has fallen behind (frames arriving slower than native),
        # resync to now instead of bursting to catch up.
        native_dt = 1.0 / streamer.effective_fps()
        now = time.monotonic()
        if next_frame > now:
            if stop_event.wait(next_frame - now):
                break
        else:
            next_frame = now
        next_frame += native_dt


def run(cfg, show=True, config_path=None, duration=None, debug=False,
        record=False, serve_api=False, node_id="node_01"):
    sc = cfg["stream"]
    dc = cfg["detector"]
    source = str(sc["source"])
    if source.isdigit():
        source = int(source)
    frame_dt = 1.0 / float(sc.get("target_fps", 25))
    idle_dt = 1.0 / float(sc.get("throttle_fps", 0.7))

    zone_cfgs = cfg.get("zones", [])
    zones = build_zones(zone_cfgs)
    counter_cfg = cfg.get("counter", {})
    audit = AuditLogger("logs/count_audit.csv") if debug else None

    streamer = VideoStreamer(source, max_queue=sc.get("queue_size", 3),
                             reconnect=sc.get("reconnect", True),
                             max_retries=sc.get("max_retries"),
                             backoff_min=sc.get("backoff_sec", 2.0),
                             gap_thresh_sec=sc.get("gap_thresh_sec", 0.5))
    det = Detector(conf=dc.get("conf", 0.20), classes=dc.get("classes"),
                   model_size=dc.get("model", "rtdetr-l"),
                   min_area=dc.get("min_area", 0), imgsz=dc.get("imgsz", 640),
                   device=dc.get("device", "cuda:0"), half=dc.get("half", True),
                   duplicate_iou=dc.get("duplicate_iou", 0.65))
    counter = ZoneCounter(zones,
                          class_labels={2: "mobil", 3: "motor", 5: "truk/bus", 7: "truk/bus"},
                          tracker_config=dc.get("tracker_config"),
                          dedup_cooldown_sec=counter_cfg.get("dedup_cooldown_sec", 4.0),
                          dedup_radius_px=counter_cfg.get("dedup_radius_px", 10),
                          min_displacement_px=counter_cfg.get("min_displacement_px", 50),
                          class_window=counter_cfg.get("class_window", 15),
                          cabin_iou=counter_cfg.get("cabin_iou", 0.6),
                          cabin_area_ratio=counter_cfg.get("cabin_area_ratio", 0.55),
                          still_count_frames=counter_cfg.get("still_count_frames", 30),
                          still_min_conf=counter_cfg.get("still_min_conf", 0.35),
                          debug=debug)
    renderer = Renderer(zones, show_dims=dc.get("show_dims", True), debug=debug)
    editor = ZoneEditor(zones, zone_cfgs)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    gpu_lock = threading.Lock()
    bus = FrameBus()
    prof = StageProfiler("single:" + str(node_id))
    reset_flag = {"v": False}

    streamer.start()
    log.info("Pipeline started. conf=%.2f imgsz=%d", det.conf, det.imgsz)

    if serve_api:
        threading.Thread(
            target=uvicorn.run,
            args=(app,),
            kwargs={"host": "0.0.0.0", "port": 8000, "log_level": "error"},
            daemon=True,
        ).start()
        log.info("API server on http://localhost:8000")

    writer = None
    audit_path = None
    audit_fps = round(1.0 / frame_dt)

    if show:
        cv2.namedWindow("ATCS", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("ATCS", editor.on_mouse)

    def sink(out):
        # Runs on the main thread: imshow/highgui stays on one thread.
        nonlocal writer, audit_path, record
        if serve_api:
            set_latest_frame(node_id, out)
        if record:
            if writer is None:
                writer, audit_path = open_audit_writer(out, audit_fps)
                if writer:
                    log.info("audit video: %s", audit_path)
                else:
                    record = False
            if writer:
                writer.write(np.ascontiguousarray(out))
        if not show:
            return
        cv2.imshow("ATCS", out)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            stop.set()
        elif k == ord("c"):
            record = not record
            if writer and not record:
                writer.release()
                log.info("audit video saved: %s", audit_path)
                writer, audit_path = None, None
            log.info("record %s", "ON" if record else "OFF")
        elif k in (ord("r"), ord("R")):
            # Defer to the AI thread; it owns the counter.
            reset_flag["v"] = True
            log.info("Counts reset")
        elif k in (ord("s"), ord("S")) and editor.dirty and config_path:
            save_zones(config_path, zones, zone_cfgs)
            editor.dirty = False
            log.info("Zones saved")

    t0 = time.monotonic()
    ai = threading.Thread(
        target=_ai_worker,
        args=(node_id, bus, streamer, det, counter, gpu_lock, frame_dt, idle_dt,
              stop, prof, reset_flag, t0, audit, debug, serve_api, True),
        daemon=True, name="ai")
    ai.start()

    deadline = t0 + duration if duration else None
    try:
        _display_worker(streamer, bus, renderer, stop, prof, sink, deadline)
    finally:
        stop.set()
        ai.join(timeout=5.0)
        streamer.stop()
        if audit:
            audit.close()
        if writer:
            writer.release()
            log.info("audit video saved: %s", audit_path)
        cv2.destroyAllWindows()
        elapsed = time.monotonic() - t0
        log.info("Final: total=%d %s", counter.total, dict(counter.total_by_class))
        log.info("Processed %d of %d delivered frames in %.0fs (%.1f fps), "
                 "dedup-suppressed=%d, connects=%d",
                 counter.n_frames, streamer.frames_delivered, elapsed,
                 counter.n_frames / max(elapsed, 1e-6), counter.suppressed,
                 streamer.connects)
    return 0


def run_multi(cfg, nodes, duration=None, debug=False):
    """Async pipeline per CCTV node, headless.

    Shared: a single Detector behind gpu_lock (one TensorRT engine in memory,
    serialized GPU inference). Per node: VideoStreamer + AI thread + display
    thread. The AI thread pushes counts via update_node_data; the display
    thread encodes the annotated frame via set_latest_frame; the frontend
    picks a node to watch and /set-focus decides which one runs full rate.
    """
    sc = cfg["stream"]
    dc = cfg["detector"]
    counter_cfg = cfg.get("counter", {})
    zone_cfgs = cfg.get("zones", [])
    frame_dt = 1.0 / float(sc.get("target_fps", 25))
    idle_dt = 1.0 / float(sc.get("throttle_fps", 0.7))

    det = Detector(conf=dc.get("conf", 0.20), classes=dc.get("classes"),
                   model_size=dc.get("model", "rtdetr-l"),
                   min_area=dc.get("min_area", 0), imgsz=dc.get("imgsz", 640),
                   device=dc.get("device", "cuda:0"), half=dc.get("half", True),
                   duplicate_iou=dc.get("duplicate_iou", 0.65))
    gpu_lock = threading.Lock()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    t_run = time.monotonic()

    threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": "0.0.0.0", "port": 8000, "log_level": "error"},
        daemon=True,
    ).start()
    log.info("API server on http://localhost:8000 (%d nodes)", len(nodes))

    workers = []
    for n in nodes:
        node_id = n["id"]
        source = n.get("source") or sc["source"]
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        zones = build_zones(n.get("zones") or zone_cfgs)
        counter = ZoneCounter(zones,
                              class_labels={2: "mobil", 3: "motor", 5: "truk/bus", 7: "truk/bus"},
                              tracker_config=dc.get("tracker_config"),
                              dedup_cooldown_sec=counter_cfg.get("dedup_cooldown_sec", 4.0),
                              dedup_radius_px=counter_cfg.get("dedup_radius_px", 10),
                              min_displacement_px=counter_cfg.get("min_displacement_px", 50),
                              class_window=counter_cfg.get("class_window", 15),
                              cabin_iou=counter_cfg.get("cabin_iou", 0.6),
                              cabin_area_ratio=counter_cfg.get("cabin_area_ratio", 0.55),
                              still_count_frames=counter_cfg.get("still_count_frames", 30),
                              still_min_conf=counter_cfg.get("still_min_conf", 0.35),
                              debug=debug)
        renderer = Renderer(zones, show_dims=dc.get("show_dims", True), debug=debug)
        streamer = VideoStreamer(source, max_queue=sc.get("queue_size", 3),
                                 reconnect=sc.get("reconnect", True),
                                 max_retries=sc.get("max_retries"),
                                 backoff_min=sc.get("backoff_sec", 2.0),
                                 gap_thresh_sec=sc.get("gap_thresh_sec", 0.5))
        streamer.start()
        bus = FrameBus()
        prof = StageProfiler("node:" + node_id)
        reset_flag = {"v": False}
        t0 = time.monotonic()

        ai = threading.Thread(
            target=_ai_worker,
            args=(node_id, bus, streamer, det, counter, gpu_lock, frame_dt,
                  idle_dt, stop, prof, reset_flag, t0, None, debug, True),
            daemon=True, name=f"ai-{node_id}")
        disp = threading.Thread(
            target=_display_worker,
            args=(streamer, bus, renderer, stop, prof,
                  lambda out, nid=node_id: set_latest_frame(nid, out)),
            daemon=True, name=f"disp-{node_id}")
        ai.start()
        disp.start()
        workers.append((node_id, streamer, ai, disp, counter))
        log.info("node %s: streaming %s", node_id, source)

    try:
        while not stop.is_set():
            if duration and time.monotonic() - t_run > duration:
                break
            if all(not ai.is_alive() and not disp.is_alive()
                   for _, _, ai, disp, _ in workers):
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop.set()
    stop.set()

    for node_id, streamer, ai, disp, counter in workers:
        ai.join(timeout=5.0)
        disp.join(timeout=5.0)
        streamer.stop()
        log.info("node %s done: counted=%d", node_id, counter.total)
    return 0


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--config", default="config/settings.yaml")
    a.add_argument("--no-display", action="store_true")
    a.add_argument("--duration", type=float, default=None)
    a.add_argument("--debug", action="store_true",
                   help="render conf scores + trajectory tails, write logs/count_audit.csv")
    a.add_argument("--record", action="store_true",
                   help="write annotated frames to logs/audit_video_*.mp4 "
                        "(works with --no-display; C toggles it in the window)")
    a.add_argument("--api", action="store_true",
                   help="serve the WebGIS API (uvicorn) in a background thread")
    a.add_argument("--multi", action="store_true",
                   help="headless worker thread per node in config/nodes.yaml "
                        "(implies API server; ignores --config stream source)")
    a.add_argument("--nodes", default=None,
                   help="comma-separated node ids to run in --multi "
                        "(default: all nodes)")
    args = a.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    setup_logging(level=cfg.get("logging", {}).get("level", "INFO"),
                  log_file=cfg.get("logging", {}).get("file", "logs/app.log"))
    if args.multi:
        from src.api.server import load_node_registry
        reg = load_node_registry()
        if args.nodes:
            keep = {n.strip() for n in args.nodes.split(",")}
            reg = [n for n in reg if n["id"] in keep]
        if not reg:
            raise SystemExit("no nodes selected (check config/nodes.yaml / --nodes)")
        sys.exit(run_multi(cfg, reg, duration=args.duration, debug=args.debug))
    sys.exit(run(cfg, show=not args.no_display, config_path=args.config,
                 duration=args.duration, debug=args.debug, record=args.record,
                 serve_api=args.api))
