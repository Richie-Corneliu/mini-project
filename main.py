"""ATCS orchestrator: settings.yaml -> stream thread -> per-frame
Grab -> Detect -> Count -> Log -> Render -> Display -> Audit video.

Every frame that arrives is processed in order: the pipeline blocks on the
streamer's bounded queue, so backpressure raises latency instead of skipping
frames (crucial for count validation against ground truth)."""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import yaml

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


def run(cfg, show=True, config_path=None, duration=None, debug=False,
        record=False):
    sc = cfg["stream"]
    dc = cfg["detector"]
    source = str(sc["source"])
    if source.isdigit():
        source = int(source)
    frame_dt = 1.0 / float(sc.get("target_fps", 25))

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
    stop = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("v", True))

    streamer.start()
    log.info("Pipeline started. conf=%.2f imgsz=%d", det.conf, det.imgsz)

    writer = None
    audit_path = None
    audit_fps = round(1.0 / frame_dt)

    if show:
        cv2.namedWindow("ATCS", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("ATCS", editor.on_mouse)

    try:
        t0 = time.monotonic()
        last_stats = t0
        next_frame = t0
        while not stop["v"]:
            if duration and time.monotonic() - t0 > duration:
                break
            item = streamer.get(timeout=1.0)
            if item is None:
                continue
            frame, arrival, wall = item
            now = time.monotonic()
            if next_frame > now:
                time.sleep(next_frame - now)
            else:
                next_frame = now
            next_frame += frame_dt
            try:
                dets = counter.track(det.detect(frame), frame)
                events = counter.update(dets, now=arrival,
                                        freeze=streamer.is_frozen())
            except Exception as e:
                log.warning("detector: %s", e)
                continue
            if audit:
                for ev in events:
                    audit.write(ev)
            if debug and events:
                for ev in events:
                    log.info("COUNT %s id=%d conf=%.2f zone=%d (total=%d)",
                             ev["class_name"], ev["track_id"], ev["confidence"],
                             ev["zone"], counter.total)

            now = time.monotonic()
            if now - last_stats >= 5.0:
                log.info("stats: frames=%d dets=%d tracked=%d counted=%d dedup=%d "
                         "cabin=%d delivered=%d connects=%d",
                         counter.n_frames, counter.n_detections, counter.n_tracked,
                         counter.total, counter.suppressed, counter.suppressed_cabin,
                         streamer.frames_delivered, streamer.connects)
                last_stats = now

            if not (show or record):
                continue
            snap = None
            if len(dets) and dets.tracker_id is not None:
                snap = (dets.xyxy, dets.tracker_id, dets.class_id,
                        centroids(dets.xyxy).astype(int), dets.confidence)
            renderer.draw(frame, snap, dict(counter.total_by_class),
                          trails=counter.trails)
            if record:
                if writer is None:
                    writer, audit_path = open_audit_writer(frame, audit_fps)
                    if writer:
                        log.info("audit video: %s", audit_path)
                    else:
                        record = False
                if writer:
                    writer.write(np.ascontiguousarray(frame))
            if not show:
                continue
            cv2.imshow("ATCS", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("c"):
                record = not record
                if writer and not record:
                    writer.release()
                    log.info("audit video saved: %s", audit_path)
                    writer, audit_path = None, None
                log.info("record %s", "ON" if record else "OFF")
            if k in (ord("r"), ord("R")):
                counter.reset()
                log.info("Counts reset")
            if k in (ord("s"), ord("S")) and editor.dirty and config_path:
                save_zones(config_path, zones, zone_cfgs)
                editor.dirty = False
                log.info("Zones saved")
    finally:
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
    args = a.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    setup_logging(level=cfg.get("logging", {}).get("level", "INFO"),
                  log_file=cfg.get("logging", {}).get("file", "logs/app.log"))
    sys.exit(run(cfg, show=not args.no_display, config_path=args.config,
                 duration=args.duration, debug=args.debug, record=args.record))
