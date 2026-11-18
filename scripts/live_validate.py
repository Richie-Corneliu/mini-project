"""Lightweight live validation audit script for a running Smart ATCS node.

Polls /api/v1/traffic-data every 200ms and records annotated frames from
/api/v1/video-feed/{node} without opening extra RTSP connections or touching
core modules.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

WIB = timezone(timedelta(hours=7))
BURST_CLASSES = ("mobil", "truk/bus")
ALL_CLASSES = ("motor", "mobil", "truk/bus")


def _load_api_key() -> str:
    key = os.getenv("ATCS_API_KEY", "").strip()
    if key:
        return key
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "ATCS_API_KEY":
                return v.strip().strip('"').strip("'")
    return ""


def _http_json(url: str, method: str = "GET", headers: dict | None = None,
               timeout: float = 5.0) -> dict:
    req = urllib.request.Request(url, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _set_focus(base_url: str, target: str, headers: dict) -> None:
    try:
        _http_json(f"{base_url}/api/v1/set-focus/{target}",
                   method="POST", headers=headers, timeout=3.0)
    except Exception:
        pass


def _extract_counts(node: dict) -> dict[str, int]:
    m = int(node.get("motor", 0))
    c = int(node.get("mobil", 0))
    t = int(node.get("truk", node.get("truk/bus", 0)))
    return {"motor": m, "mobil": c, "truk/bus": t, "total": m + c + t}


class _MjpegRecorder:
    """Reads /api/v1/video-feed/{node} and writes unique frames to MP4."""

    def __init__(self, feed_url: str, video_path: Path, target_fps: float):
        self.feed_url = feed_url
        self.video_path = video_path
        self.target_fps = max(5.0, min(30.0, float(target_fps or 15.0)))
        self.stop = threading.Event()
        self.frames_received = 0
        self.frames_written = 0
        self.duplicates_skipped = 0
        self.error: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="mjpeg-rec")

    def start(self) -> None:
        self._thread.start()

    def close(self, timeout: float = 5.0) -> None:
        self.stop.set()
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        writer = None
        last_jpg = None
        buf = bytearray()
        try:
            req = urllib.request.Request(self.feed_url)
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                while not self.stop.is_set():
                    chunk = resp.read(32768)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    while not self.stop.is_set():
                        soi = buf.find(b"\xff\xd8")
                        if soi < 0:
                            if len(buf) > 131072:
                                del buf[:-4]
                            break
                        if soi > 0:
                            del buf[:soi]
                        eoi = buf.find(b"\xff\xd9", 2)
                        if eoi < 0:
                            break
                        jpg = bytes(buf[: eoi + 2])
                        del buf[: eoi + 2]
                        self.frames_received += 1
                        if jpg == last_jpg:
                            self.duplicates_skipped += 1
                            continue
                        last_jpg = jpg
                        arr = np.frombuffer(jpg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is None:
                            continue
                        if writer is None:
                            h, w = frame.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            writer = cv2.VideoWriter(
                                str(self.video_path), fourcc,
                                round(self.target_fps), (w, h),
                            )
                            if not writer.isOpened():
                                writer.release()
                                writer = None
                                self.error = f"VideoWriter failed for {self.video_path}"
                                return
                        writer.write(frame)
                        self.frames_written += 1
        except Exception as exc:
            if not self.stop.is_set():
                self.error = str(exc)
        finally:
            if writer is not None:
                writer.release()


def run_audit(node_id: str, duration: float, base_url: str,
              poll_interval: float, out_dir: Path, manage_focus: bool) -> int:
    base_url = base_url.rstrip("/")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_tag = datetime.now(WIB).strftime("%Y%m%d_%H%M%S")
    video_path = out_dir / f"audit_{node_id}_{ts_tag}.mp4"
    json_path = out_dir / f"audit_{node_id}_{ts_tag}.json"

    api_key = _load_api_key()
    auth_headers = {"X-API-Key": api_key} if api_key else {}

    initial_payload = _http_json(f"{base_url}/api/v1/traffic-data", timeout=5.0)
    nodes_map = initial_payload.get("nodes") or {}
    if node_id not in nodes_map:
        raise SystemExit(f"Node '{node_id}' not found in /api/v1/traffic-data")
    init_node = nodes_map[node_id]
    node_nama = init_node.get("nama", node_id)
    init_fps = float(init_node.get("fps") or 15.0)

    if manage_focus:
        _set_focus(base_url, node_id, auth_headers)

    recorder = _MjpegRecorder(
        f"{base_url}/api/v1/video-feed/{node_id}",
        video_path,
        target_fps=init_fps if init_fps >= 5.0 else 15.0,
    )
    recorder.start()

    start_iso = datetime.now(WIB).isoformat(timespec="seconds")
    t0 = time.monotonic()
    start_counts = _extract_counts(init_node)
    prev_counts = dict(start_counts)
    end_counts = dict(start_counts)

    recent_hits: dict[str, deque[float]] = {c: deque() for c in BURST_CLASSES}
    events = []
    anomalies = []
    fps_samples = []
    status_counter: Counter[str] = Counter()
    poll_samples = 0
    poll_errors = 0

    try:
        next_poll = t0
        while True:
            now_mono = time.monotonic()
            elapsed = now_mono - t0
            if elapsed >= duration:
                break
            sleep_for = next_poll - now_mono
            if sleep_for > 0:
                time.sleep(min(sleep_for, max(0.0, duration - elapsed)))
                now_mono = time.monotonic()
                elapsed = now_mono - t0
                if elapsed >= duration:
                    break
            next_poll += poll_interval
            if next_poll < now_mono:
                next_poll = now_mono + poll_interval

            try:
                payload = _http_json(f"{base_url}/api/v1/traffic-data", timeout=3.0)
            except Exception as exc:
                poll_errors += 1
                anomalies.append({
                    "type": "poll_error",
                    "elapsed_sec": round(elapsed, 2),
                    "timestamp": datetime.now(WIB).isoformat(timespec="milliseconds"),
                    "detail": str(exc),
                })
                continue

            node = (payload.get("nodes") or {}).get(node_id)
            if node is None:
                continue
            poll_samples += 1
            status = str(node.get("stream_status") or "UNKNOWN")
            status_counter[status] += 1
            fps = float(node.get("fps") or 0.0)
            if fps > 0:
                fps_samples.append(fps)

            cur = _extract_counts(node)
            end_counts = cur
            deltas = {c: cur[c] - prev_counts[c] for c in ALL_CLASSES}
            if any(deltas[c] != 0 for c in ALL_CLASSES):
                ts_now = datetime.now(WIB).isoformat(timespec="milliseconds")
                events.append({
                    "elapsed_sec": round(elapsed, 2),
                    "timestamp": ts_now,
                    "delta": deltas,
                    "counts": cur,
                    "fps": fps,
                    "occupancy": int(node.get("occupancy") or 0),
                    "stream_status": status,
                })
                for cls_name in BURST_CLASSES:
                    d = deltas[cls_name]
                    if d < 0:
                        anomalies.append({
                            "type": "counter_drop",
                            "class_name": cls_name,
                            "delta": d,
                            "elapsed_sec": round(elapsed, 2),
                            "timestamp": ts_now,
                            "counts": cur,
                        })
                        recent_hits[cls_name].clear()
                        continue
                    if d > 0:
                        q = recent_hits[cls_name]
                        for _ in range(d):
                            q.append(now_mono)
                        while q and (now_mono - q[0]) > 0.42:
                            q.popleft()
                        if len(q) >= 2:
                            anomalies.append({
                                "type": "rapid_double_count_burst",
                                "class_name": cls_name,
                                "hits_in_window": len(q),
                                "window_sec": 0.4,
                                "delta": d,
                                "elapsed_sec": round(elapsed, 2),
                                "timestamp": ts_now,
                                "counts": cur,
                            })
            prev_counts = cur
    finally:
        recorder.close()
        if manage_focus:
            _set_focus(base_url, "none", auth_headers)

    actual_duration = max(time.monotonic() - t0, 1e-6)
    end_iso = datetime.now(WIB).isoformat(timespec="seconds")
    net_delta = {k: end_counts[k] - start_counts[k] for k in ("motor", "mobil", "truk/bus", "total")}
    hourly_scale = 3600.0 / actual_duration
    hourly_rate = {k: round(net_delta[k] * hourly_scale, 1) for k in ("motor", "mobil", "truk/bus", "total")}

    report = {
        "node_id": node_id,
        "node_nama": node_nama,
        "base_url": base_url,
        "started_at": start_iso,
        "ended_at": end_iso,
        "duration_sec": round(actual_duration, 2),
        "poll_interval_sec": poll_interval,
        "video_file": str(video_path) if video_path.exists() else None,
        "recorder_error": recorder.error,
        "start_counts": start_counts,
        "end_counts": end_counts,
        "net_delta": net_delta,
        "effective_hourly_rate": hourly_rate,
        "stream_health": {
            "poll_samples": poll_samples,
            "poll_errors": poll_errors,
            "status_counts": dict(status_counter),
            "avg_fps": round(sum(fps_samples) / len(fps_samples), 2) if fps_samples else 0.0,
            "min_fps": round(min(fps_samples), 2) if fps_samples else 0.0,
            "max_fps": round(max(fps_samples), 2) if fps_samples else 0.0,
            "frames_received": recorder.frames_received,
            "frames_written": recorder.frames_written,
            "duplicates_skipped": recorder.duplicates_skipped,
        },
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "events": events,
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"=== LIVE AUDIT SUMMARY: {node_id} ({node_nama}) ===")
    print(f"Duration   : {actual_duration:.1f}s | Samples: {poll_samples} | Errors: {poll_errors}")
    print(f"Stream FPS : avg={report['stream_health']['avg_fps']} "
          f"(min={report['stream_health']['min_fps']}, max={report['stream_health']['max_fps']}) | "
          f"Status={dict(status_counter)}")
    print(f"Video      : {report['video_file']} (written={recorder.frames_written}, "
          f"dups_skipped={recorder.duplicates_skipped})")
    print(f"JSON Log   : {json_path}")
    print("-" * 64)
    print(f"{'Kelas':<12} {'Start':>8} {'End':>8} {'Delta':>8} {'Rate/jam':>10}")
    for k in ("motor", "mobil", "truk/bus", "total"):
        print(f"{k:<12} {start_counts[k]:>8} {end_counts[k]:>8} "
              f"{net_delta[k]:>+8} {hourly_rate[k]:>10.1f}")
    print("-" * 64)
    print(f"Anomalies  : {len(anomalies)}")
    for a in anomalies[:10]:
        print(f"  - [{a['type']}] {a.get('class_name', '')} @ +{a['elapsed_sec']}s {a.get('counts', '')}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Live HTTP+MJPEG validation audit")
    p.add_argument("--node", default="node_02")
    p.add_argument("--duration", type=float, default=180.0)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--poll-interval", type=float, default=0.2)
    p.add_argument("--out-dir", default="logs/benchmarks")
    p.add_argument("--no-set-focus", action="store_true",
                   help="do not call /api/v1/set-focus before/after capture")
    args = p.parse_args()
    return run_audit(
        node_id=args.node,
        duration=args.duration,
        base_url=args.base_url,
        poll_interval=args.poll_interval,
        out_dir=Path(args.out_dir),
        manage_focus=not args.no_set_focus,
    )


if __name__ == "__main__":
    raise SystemExit(main())
