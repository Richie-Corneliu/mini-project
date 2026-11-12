"""Headless production-pipeline benchmark for recorded video or frame iterables."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import yaml

try:
    from scripts.evaluate_accuracy import (compute_accuracy_report,
                                            format_markdown, load_counts,
                                            parse_cli_counts)
except ImportError:  # python scripts/benchmark_video.py
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.evaluate_accuracy import (compute_accuracy_report,
                                            format_markdown, load_counts,
                                            parse_cli_counts)

from src.core.detector import Detector
from src.core.zone_counter import ZoneCounter
from src.ui.renderer import Renderer

ROOT = Path(__file__).resolve().parents[1]
COUNTER_LABELS = {2: "mobil", 3: "motor", 5: "truk/bus", 7: "truk/bus"}
NODE_COUNTER_KEYS = ("dedup_cooldown_sec", "dedup_radius_px", "min_displacement_px",
                     "class_window", "motor_lock_frames", "motor_flip_frames",
                     "car_lock_frames", "car_flip_frames", "cabin_iou",
                     "cabin_area_ratio", "still_count_frames", "still_min_conf",
                     "ghost_ttl_frames", "ghost_radius_px")


def resolve_node_config(node_id=None, *, settings_path="config/settings.yaml",
                        nodes_path="config/nodes.yaml", overrides=None):
    """Return main-compatible settings with selected node and CLI overrides applied."""
    settings_path = Path(settings_path)
    if not settings_path.is_absolute():
        settings_path = ROOT / settings_path
    cfg = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    if node_id:
        nodes_path = Path(nodes_path)
        if not nodes_path.is_absolute():
            nodes_path = ROOT / nodes_path
        entries = (yaml.safe_load(nodes_path.read_text(encoding="utf-8")) or {}).get("nodes", [])
        node = next((n for n in entries if n.get("id") == node_id), None)
        if node is None:
            raise ValueError(f"unknown node: {node_id}")
        cfg["node"] = node
        cfg["zones"] = node.get("zones") or cfg.get("zones", [])
        cfg["counter"] = {**cfg.get("counter", {}),
                           **{k: node[k] for k in NODE_COUNTER_KEYS if k in node}}
        cfg["detector"] = {**cfg.get("detector", {}),
                            **{k: node[k] for k in ("car_min_area", "truck_min_area")
                               if k in node}}
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        section, name = (("counter", key) if key in NODE_COUNTER_KEYS
                         else ("detector", key))
        cfg.setdefault(section, {})[name] = value
    return cfg


def _video_frames(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise OSError(f"cannot open video: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def _area_summary(values):
    if not values:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
    return {"count": len(values), "min": int(min(values)),
            "median": float(statistics.median(values)),
            "p90": float(np.percentile(values, 90)), "max": int(max(values))}


def run_video_benchmark(source, detector, counter, *, car_min_area=None,
                        truck_min_area=None, conf=None, max_frames=None,
                        stride=1, save_video=None, renderer=None, gt=None,
                        scenario="", output_dir="logs/benchmarks", save_report=False):
    """Run Detector -> ZoneCounter.track -> ZoneCounter.update headlessly."""
    if stride < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError("stride and max_frames must be positive")
    frames = _video_frames(source) if isinstance(source, (str, Path)) else iter(source)
    areas = defaultdict(list)
    processed = seen = 0
    start = time.perf_counter()
    writer = None
    has_conf = hasattr(detector, "conf")
    detector_conf = getattr(detector, "conf", None)
    if conf is not None and has_conf:
        detector.conf = conf
    try:
        for frame in frames:
            if seen % stride:
                seen += 1
                continue
            seen += 1
            detect = getattr(detector, "detect", detector)
            raw = detect(frame, car_min_area=car_min_area,
                         truck_min_area=truck_min_area)
            tracked = counter.track(raw, frame) if hasattr(counter, "track") else raw
            counter.update(tracked)
            processed += 1
            if tracked is not None and len(tracked) and tracked.tracker_id is not None:
                for i, tid in enumerate(tracked.tracker_id.astype(int)):
                    diag = counter.get_track_diagnostics(int(tid), tracked.xyxy[i]) \
                        if hasattr(counter, "get_track_diagnostics") else {}
                    cls = int(tracked.class_id[i]) if tracked.class_id is not None else -1
                    label = COUNTER_LABELS.get(cls, str(cls))
                    area = diag.get("area_px")
                    if area is not None:
                        areas[label].append(area)
            if save_video:
                if renderer is None:
                    renderer = Renderer(counter.zones, show_diagnostics=True,
                                        counter=counter, car_min_area=car_min_area,
                                        truck_min_area=truck_min_area)
                if writer is None:
                    h, w = frame.shape[:2]
                    path = Path(save_video)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             25.0, (w, h))
                    if not writer.isOpened():
                        writer.release()
                        raise OSError(f"cannot open video writer: {path}")
                snap = None
                if len(tracked) and tracked.tracker_id is not None:
                    cents = ((tracked.xyxy[:, :2] + tracked.xyxy[:, 2:]) / 2).astype(int)
                    snap = (tracked.xyxy, tracked.tracker_id, tracked.class_id,
                            cents, tracked.confidence)
                annotated = frame.copy()
                renderer.draw(annotated, snap, dict(counter.total_by_class))
                writer.write(annotated)
            if max_frames is not None and processed >= max_frames:
                break
    finally:
        if writer is not None:
            writer.release()
        close = getattr(frames, "close", None)
        if close:
            close()
        if conf is not None and has_conf:
            detector.conf = detector_conf
    elapsed = time.perf_counter() - start
    pred = {"motor": int(counter.total_by_class.get("motor", 0)),
            "mobil": int(counter.total_by_class.get("mobil", 0)),
            "truk_bus": int(counter.total_by_class.get("truk/bus", 0))}
    report = {"scenario": scenario, "frames_processed": processed,
              "elapsed_seconds": elapsed,
              "fps": processed / elapsed if elapsed else 0.0,
              "counts": pred, "total": int(counter.total),
              "ghost_handovers": int(getattr(counter, "ghost_handovers", 0)),
              "suppressed": int(getattr(counter, "suppressed", 0)),
              "area_px_by_class": {k: _area_summary(v) for k, v in areas.items()}}
    if gt is not None:
        accuracy = compute_accuracy_report(gt, pred, scenario=scenario)
        report["accuracy"] = accuracy
        report["accuracy_markdown"] = format_markdown(accuracy)
        print(report["accuracy_markdown"])
    print(f"FPS: {report['fps']:.2f} | ghost handovers: {report['ghost_handovers']} | "
          f"suppressed: {report['suppressed']} | areas: "
          f"{json.dumps(report['area_px_by_class'], sort_keys=True)}")
    if save_report or gt is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = out / f"video_benchmark_{stamp}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        md = [f"# Video Benchmark - {scenario or 'Recorded video'}", "",
              f"Frames: {processed} | FPS: {report['fps']:.2f}",
              f"Counts: {json.dumps(pred, ensure_ascii=False)}", "",
              f"Ghost handovers: {report['ghost_handovers']} | Suppressed: {report['suppressed']}",
              "", "## Area calibration (px²)", "", "```json",
              json.dumps(report["area_px_by_class"], indent=2), "```"]
        if gt is not None:
            md += ["", report["accuracy_markdown"]]
        path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
        report["json_path"] = str(path)
        report["markdown_path"] = str(path.with_suffix(".md"))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video")
    p.add_argument("--config", default="config/settings.yaml")
    p.add_argument("--nodes", default="config/nodes.yaml")
    p.add_argument("--node")
    p.add_argument("--scenario", default="")
    p.add_argument("--car-min-area", type=float)
    p.add_argument("--truck-min-area", type=float)
    p.add_argument("--min-displacement-px", type=float)
    p.add_argument("--ghost-ttl-frames", type=int)
    p.add_argument("--ghost-radius-px", type=float)
    p.add_argument("--conf", type=float)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--save-video")
    p.add_argument("--gt")
    p.add_argument("--gt-counts", help="Motor,Mobil,Truk/Bus")
    p.add_argument("--output-dir", default="logs/benchmarks")
    p.add_argument("--save-report", action="store_true")
    a = p.parse_args(argv)
    cfg = resolve_node_config(a.node, settings_path=a.config, nodes_path=a.nodes,
                              overrides={"car_min_area": a.car_min_area,
                                         "truck_min_area": a.truck_min_area,
                                         "min_displacement_px": a.min_displacement_px,
                                         "ghost_ttl_frames": a.ghost_ttl_frames,
                                         "ghost_radius_px": a.ghost_radius_px,
                                         "conf": a.conf})
    from main import build_zones
    zones = build_zones(cfg.get("zones", []))
    dc, cc = cfg["detector"], cfg.get("counter", {})
    detector = Detector(conf=dc.get("conf", .2), classes=dc.get("classes"),
                        model_size=dc.get("model", "rtdetr-l"), min_area=dc.get("min_area", 0),
                        imgsz=dc.get("imgsz", 640), device=dc.get("device", "cuda:0"),
                        half=dc.get("half"), duplicate_iou=dc.get("duplicate_iou", .65),
                        truck_min_area=dc.get("truck_min_area", 10500),
                        car_min_area=dc.get("car_min_area", 3000))
    counter = ZoneCounter(zones, class_labels=COUNTER_LABELS,
                          tracker_config=dc.get("tracker_config"), **cc)
    gt = load_counts(a.gt) if a.gt else parse_cli_counts(a.gt_counts) if a.gt_counts else None
    run_video_benchmark(a.video, detector, counter,
                        car_min_area=dc.get("car_min_area"),
                        truck_min_area=dc.get("truck_min_area"), conf=a.conf,
                        max_frames=a.max_frames, stride=a.stride,
                        save_video=a.save_video, gt=gt, scenario=a.scenario,
                        output_dir=a.output_dir, save_report=a.save_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
