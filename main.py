"""ATCS orchestrator: settings.yaml -> stream thread -> per-frame
Grab -> Detect -> Count -> Log -> Render -> Display -> Audit video.

Asynchronous pipeline, three threads per node:
  ingest  VideoStreamer decodes HLS/RTSP into a bounded queue (native rate)
  display consumes every frame, draws the newest overlay, shows / publishes it
          (render + MJPEG encode are skipped for nodes nobody is watching)
  ai      samples every raw frame the display delivers (acquire rate, no
          throttle) and runs RT-DETR + ByteTrack; counting accuracy must not
          depend on whether an operator is watching

The display loop never waits on the model, so the video stays fluid at the
source's native rate while the boxes refresh at the AI rate. The display is
the only consumer of the stream queue: the AI samples the shared newest frame
instead, otherwise the two would split frames ~50/50 and neither would hit
its target rate."""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import uvicorn
import yaml

from src.api.server import (_expand_source, app, clear_commands,
                             get_active_focus, mark_node_reconnecting,
                             pop_command, register_stats_store,
                             register_zone_update_hook, seed_node_data,
                             set_latest_frame, set_node_frame_size,
                             update_node_data)
from src.core.detector import Detector, centroids
from src.core.zone_counter import ZoneCounter
from src.stream.video_streamer import VideoStreamer, redact_url
from src.ui.renderer import Renderer, ZoneEditor
from src.utils import stats_store as _stats_mod
from src.utils.logger import AuditLogger, get_logger, setup_logging
from src.utils.stats_store import StatsStore, fold_slot_counts, seed_counter
from src.utils.zone_store import persist_zone

log = get_logger(__name__)


def build_zones(zone_cfgs):
    zones = []
    for zc in zone_cfgs:
        z = sv.PolygonZone(polygon=np.array(zc["polygon"], dtype=np.float32))
        if "allowed_directions" in zc:
            z.allowed_directions = zc["allowed_directions"]
        if "flow_max_angle_deg" in zc:
            z.flow_max_angle_deg = zc["flow_max_angle_deg"]
        zones.append(z)
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


def _seed_api_from_stats(stats):
    """Publish the restored snapshot before the first frame arrives.

    A camera still warming up (or offline) would otherwise show zeros on the
    dashboard until its worker publishes, hiding counts already on disk.
    """
    for node_id, node in stats.snapshot()["nodes"].items():
        seed_node_data(node_id, {
            "motor": node["counts"]["motor"],
            "mobil": node["counts"]["mobil"],
            "truk": node["counts"]["truk"],
            "hourly_counts": node["hourly_counts"],
            "slot_counts": node.get("slot_counts", {}),
            "session_starts": node.get("session_starts", []),
        })


def _build_detector(cfg, polygons=None):
    dc = cfg["detector"]
    return Detector(conf=dc.get("conf_threshold", dc.get("conf", 0.20)),
                    classes=dc.get("classes"),
                    model_size=dc.get("model", "rtdetr-l"),
                    min_area=dc.get("min_area", 0), imgsz=dc.get("imgsz", 640),
                    device=dc.get("device", "cuda:0"), half=dc.get("half", None),
                    duplicate_iou=dc.get("duplicate_iou", 0.65),
                    truck_min_area=dc.get("truck_min_area", 12000),
                    car_min_area=dc.get("car_min_area", 3000),
                    truck_min_height_ratio=dc.get("truck_min_height_ratio", 0.070),
                    perspective_scaling=dc.get("perspective_scaling", True),
                    y_ref_far=dc.get("y_ref_far"),
                    y_ref_mid=dc.get("y_ref_mid"),
                    y_ref_near=dc.get("y_ref_near"),
                    scale_far=dc.get("scale_far", 0.42),
                    scale_near=dc.get("scale_near", 1.95),
                    polygons=polygons)


def _build_pipeline(cfg, zones, source, debug):
    """Per-node streamer + counter + renderer from config, shared by run()
    and run_multi(). `source` is this node's raw URL or USB device index."""
    sc = cfg["stream"]
    dc = cfg["detector"]
    counter_cfg = cfg.get("counter", {})
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    streamer = VideoStreamer(source, max_queue=sc.get("queue_size", 3),
                             reconnect=sc.get("reconnect", True),
                             max_retries=sc.get("max_retries"),
                             backoff_min=sc.get("backoff_sec", 2.0),
                             gap_thresh_sec=sc.get("gap_thresh_sec", 0.5))
    counter = ZoneCounter(zones,
                          class_labels={2: "mobil", 3: "motor", 5: "truk/bus", 7: "truk/bus"},
                          tracker_config=dc.get("tracker_config"),
                          dedup_cooldown_sec=counter_cfg.get("dedup_cooldown_sec", 4.0),
                          dedup_radius_px=counter_cfg.get("dedup_radius_px", 10),
                          min_displacement_px=counter_cfg.get("min_displacement_px", 12),
                          class_window=counter_cfg.get("class_window", 15),
                          motor_lock_frames=counter_cfg.get("motor_lock_frames", 3),
                          motor_flip_frames=counter_cfg.get("motor_flip_frames", 4),
                          car_lock_frames=counter_cfg.get("car_lock_frames", 3),
                          car_flip_frames=counter_cfg.get("car_flip_frames", 6),
                          cabin_iou=counter_cfg.get("cabin_iou", 0.6),
                          cabin_area_ratio=counter_cfg.get("cabin_area_ratio", 0.55),
                          still_count_frames=counter_cfg.get("still_count_frames", 30),
                          still_min_conf=counter_cfg.get("still_min_conf", 0.35),
                          ghost_ttl_frames=counter_cfg.get("ghost_ttl_frames", 12),
                          ghost_radius_px=counter_cfg.get("ghost_radius_px", 35),
                          max_stale_frames=counter_cfg.get("max_stale_frames", 300),
                          allowed_directions=counter_cfg.get("allowed_directions"),
                          flow_max_angle_deg=counter_cfg.get("flow_max_angle_deg", 75.0),
                          min_track_frames=counter_cfg.get("min_track_frames", 2),
                          zone_dwell_frames=counter_cfg.get("zone_dwell_frames", 1),
                          truck_min_area=counter_cfg.get("truck_min_area", dc.get("truck_min_area", 12000)),
                          truck_confirm_frames=counter_cfg.get("truck_confirm_frames", 5),
                          truck_min_height_ratio=counter_cfg.get(
                              "truck_min_height_ratio", dc.get("truck_min_height_ratio", 0.070)
                          ),
                          scale_far=dc.get("scale_far", 0.42),
                          scale_near=dc.get("scale_near", 1.95),
                          y_ref_far=dc.get("y_ref_far"),
                          y_ref_mid=dc.get("y_ref_mid"),
                          y_ref_near=dc.get("y_ref_near"),
                          debug=debug)
    renderer = Renderer(zones, show_dims=dc.get("show_dims", True), debug=debug,
                        show_diagnostics=cfg.get("ui", {}).get("show_diagnostics"),
                        counter=counter,
                        car_min_area=dc.get("car_min_area"),
                        truck_min_area=counter_cfg.get("truck_min_area", dc.get("truck_min_area")))
    return streamer, counter, renderer


def _polygon_signature(zones_cfg):
    """Canonical tuple of polygon vertices and flow rules across zones."""
    sig = []
    for z in zones_cfg or []:
        if hasattr(z, "polygon"):
            pts = np.asarray(z.polygon, dtype=float).reshape(-1, 2).tolist()
            ad_raw = getattr(z, "allowed_directions", None)
            ang_raw = getattr(z, "flow_max_angle_deg", None)
        elif isinstance(z, dict):
            pts = [[float(p[0]), float(p[1])] for p in (z.get("polygon") or [])]
            ad_raw = z.get("allowed_directions")
            ang_raw = z.get("flow_max_angle_deg")
        else:
            pts = [[float(p[0]), float(p[1])] for p in z]
            ad_raw = None
            ang_raw = None
        ad_sig = (
            tuple(
                tuple(float(v) for v in d) if isinstance(d, (list, tuple)) else str(d)
                for d in ad_raw
            )
            if ad_raw is not None else None
        )
        ang_sig = None if ang_raw is None else float(ang_raw)
        sig.append((tuple((round(x, 3), round(y, 3)) for x, y in pts), ad_sig, ang_sig))
    return tuple(sig)


_ACTIVE_RELOADER_LOCK = threading.Lock()
_ACTIVE_RELOADER: ConfigHotReloader | None = None


def apply_node_zones(node_id: str, zones,
                     frame_width: int | None = None,
                     frame_height: int | None = None) -> bool:
    """Thread-safe callback to update a live node's ZoneCounter, Renderer, and Detector polygons."""
    with _ACTIVE_RELOADER_LOCK:
        reloader = _ACTIVE_RELOADER
    if reloader is None:
        return False
    return reloader.apply_node_zones(
        node_id, zones, frame_width=frame_width, frame_height=frame_height
    )


class ConfigHotReloader:
    """Poll settings.yaml and nodes.yaml mtimes and reload thresholds live."""

    def __init__(self, settings_path="config/settings.yaml",
                 nodes_path="config/nodes.yaml",
                 det=None, gpu_lock=None,
                 check_interval_sec: float = 2.0):
        self.settings_path = Path(settings_path) if settings_path else None
        self.nodes_path = Path(nodes_path) if nodes_path else None
        self.det = det
        self.gpu_lock = gpu_lock
        self.check_interval_sec = max(0.0, float(check_interval_sec))
        self._lock = threading.Lock()
        self._last_check = -self.check_interval_sec
        self._mtimes = self._read_mtimes()
        self._version = 0
        self._latest_settings = None
        self._latest_nodes = {}
        self._nodes = {}
        self._last_node_polygons = {}

    def register_node(self, node_id, counter, renderer=None, det_overrides=None):
        with self._lock:
            key = str(node_id)
            self._nodes[key] = [counter, renderer, det_overrides, self._version]
            self._last_node_polygons[key] = _polygon_signature(
                counter.zones if counter is not None else []
            )

    def apply_node_zones(self, node_id: str, zones,
                         frame_width: int | None = None,
                         frame_height: int | None = None) -> bool:
        """Update ZoneCounter, Renderer, and Detector polygons in-place for node_id."""
        key = str(node_id)
        with self._lock:
            entry = self._nodes.get(key)
            if entry is None:
                return False
            counter, renderer, det_overrides, _ = entry

        if counter is None:
            return False

        if (zones and isinstance(zones[0], (list, tuple, np.ndarray))
                and not isinstance(zones[0], dict)
                and np.asarray(zones[0], dtype=float).ndim == 1):
            fw = int(frame_width or 1)
            fh = int(frame_height or 1)
            counter.update_polygon(zones, fw, fh, zone_index=0, preserve_zone_attrs=True)
        else:
            zone_list = list(zones or [])
            if not zone_list:
                return False
            if len(zone_list) == len(counter.zones) == 1:
                z0 = zone_list[0]
                if isinstance(z0, dict):
                    pts0 = z0["polygon"]
                    ad0 = z0.get("allowed_directions")
                    ang0 = z0.get("flow_max_angle_deg")
                else:
                    pts0 = z0.polygon
                    ad0 = getattr(z0, "allowed_directions", None)
                    ang0 = getattr(z0, "flow_max_angle_deg", None)
                fw = int(frame_width or 1)
                fh = int(frame_height or 1)
                counter.update_polygon(
                    pts0, fw, fh, zone_index=0,
                    allowed_directions=ad0, flow_max_angle_deg=ang0,
                )
            else:
                counter.set_zones(zone_list)

        if renderer is not None:
            renderer.update_zones(counter.zones)

        def _update_det():
            if det_overrides is not None:
                det_overrides["polygons"] = list(counter.zones)
            if self.det is not None:
                self.det.configure_perspective_from_polygons(counter.zones)

        if self.gpu_lock is not None:
            with self.gpu_lock:
                _update_det()
        else:
            _update_det()

        with self._lock:
            self._last_node_polygons[key] = _polygon_signature(counter.zones)
        log.info("Applied live zone polygon update for node %s", key)
        return True

    @staticmethod
    def _safe_mtime(path: Path | None):
        if path is None:
            return None
        try:
            return os.path.getmtime(path)
        except OSError:
            return None

    def _read_mtimes(self):
        return (self._safe_mtime(self.settings_path),
                self._safe_mtime(self.nodes_path))

    @staticmethod
    def _safe_load_yaml(path: Path | None):
        if path is None or not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            log.warning("hot-reload failed reading %s (%s)", path, e)
            return None

    def _apply_detector(self, dc: dict) -> None:
        if self.det is None:
            return

        def _do():
            if "car_min_area" in dc:
                self.det.car_min_area = float(dc["car_min_area"])
            if "truck_min_area" in dc:
                self.det.truck_min_area = float(dc["truck_min_area"])
            if "truck_min_height_ratio" in dc:
                self.det.truck_min_height_ratio = max(0.0, float(dc["truck_min_height_ratio"]))
            if "scale_far" in dc:
                self.det.scale_far = float(dc["scale_far"])
            if "scale_near" in dc:
                self.det.scale_near = float(dc["scale_near"])
            if "perspective_scaling" in dc:
                self.det.perspective_scaling = bool(dc["perspective_scaling"])
            conf_val = dc.get("conf_threshold", dc.get("conf"))
            if conf_val is not None:
                self.det.conf = float(conf_val)

        if self.gpu_lock is not None:
            with self.gpu_lock:
                _do()
        else:
            _do()

    @staticmethod
    def _apply_node_cfg(node_id, counter, renderer, det_overrides,
                        settings_cfg, nodes_map) -> None:
        dc = settings_cfg.get("detector", {}) if settings_cfg else {}
        cc = settings_cfg.get("counter", {}) if settings_cfg else {}
        node_cfg = nodes_map.get(str(node_id), {})

        if det_overrides is not None:
            for key in ("car_min_area", "truck_min_area", "truck_min_height_ratio",
                        "scale_far", "scale_near",
                        "perspective_scaling", "y_ref_far", "y_ref_mid", "y_ref_near"):
                if key in node_cfg and node_cfg[key] is not None:
                    det_overrides[key] = (
                        bool(node_cfg[key]) if key == "perspective_scaling"
                        else float(node_cfg[key])
                    )
                else:
                    det_overrides.pop(key, None)

        merged_cc = {**cc}
        for key in ("min_displacement_px", "allowed_directions", "flow_max_angle_deg",
                    "ghost_ttl_frames", "ghost_radius_px", "truck_confirm_frames",
                    "truck_min_height_ratio", "min_track_frames"):
            if key in node_cfg:
                merged_cc[key] = node_cfg[key]

        effective_truck = node_cfg.get("truck_min_area", dc.get("truck_min_area"))
        effective_car = node_cfg.get("car_min_area", dc.get("car_min_area"))
        effective_h_ratio = node_cfg.get("truck_min_height_ratio", dc.get("truck_min_height_ratio"))
        effective_scale_far = node_cfg.get("scale_far", dc.get("scale_far"))
        effective_scale_near = node_cfg.get("scale_near", dc.get("scale_near"))

        if counter is not None:
            try:
                counter.update_thresholds(
                    min_displacement_px=merged_cc.get("min_displacement_px"),
                    allowed_directions=merged_cc.get("allowed_directions"),
                    flow_max_angle_deg=merged_cc.get("flow_max_angle_deg"),
                    ghost_ttl_frames=merged_cc.get("ghost_ttl_frames"),
                    ghost_radius_px=merged_cc.get("ghost_radius_px"),
                    truck_min_area=effective_truck,
                    truck_confirm_frames=merged_cc.get("truck_confirm_frames"),
                    truck_min_height_ratio=effective_h_ratio,
                    scale_far=effective_scale_far,
                    scale_near=effective_scale_near,
                    min_track_frames=merged_cc.get("min_track_frames"),
                )
            except ValueError as e:
                log.warning("hot-reload rejected counter config for %s (%s)", node_id, e)

        if renderer is not None:
            if effective_car is not None:
                renderer.car_min_area = float(effective_car)
            if effective_truck is not None:
                renderer.truck_min_area = float(effective_truck)

    def check_and_reload(self, now: float | None = None,
                         target_node_id: str | None = None,
                         apply_all: bool | None = None) -> bool:
        if apply_all is None:
            apply_all = (target_node_id is None)
        current = time.monotonic() if now is None else float(now)
        reloaded_file = False
        with self._lock:
            if current - self._last_check >= self.check_interval_sec:
                self._last_check = current
                mtimes = self._read_mtimes()
                if mtimes != self._mtimes:
                    settings_cfg = self._safe_load_yaml(self.settings_path)
                    nodes_raw = self._safe_load_yaml(self.nodes_path)
                    if settings_cfg is not None or nodes_raw is not None:
                        if settings_cfg is not None:
                            self._latest_settings = settings_cfg
                            self._apply_detector(settings_cfg.get("detector", {}))
                        if nodes_raw is not None:
                            entries = nodes_raw.get("nodes") or []
                            self._latest_nodes = {
                                str(n["id"]): n for n in entries
                                if isinstance(n, dict) and "id" in n
                            }
                        self._mtimes = mtimes
                        self._version += 1
                        reloaded_file = True
                        log.info("Hot-reloaded config thresholds (v%d)", self._version)

            version = self._version
            settings_cfg = self._latest_settings
            nodes_map = self._latest_nodes
            targets = []
            if apply_all:
                for nid, entry in self._nodes.items():
                    if entry[3] < version:
                        entry[3] = version
                        targets.append((nid, entry[0], entry[1], entry[2]))
            else:
                key = str(target_node_id)
                if key in self._nodes:
                    entry = self._nodes[key]
                    if entry[3] < version:
                        entry[3] = version
                        targets.append((key, entry[0], entry[1], entry[2]))

        for nid, counter, renderer, det_over in targets:
            self._apply_node_cfg(nid, counter, renderer, det_over, settings_cfg, nodes_map)
            node_cfg = nodes_map.get(str(nid), {})
            zones_cfg = node_cfg.get("zones")
            if zones_cfg:
                with self._lock:
                    prev_sig = self._last_node_polygons.get(str(nid))
                if _polygon_signature(zones_cfg) != prev_sig:
                    self.apply_node_zones(str(nid), zones_cfg)
        return reloaded_file or bool(targets)


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


def _ai_worker(node_id, bus, streamer, det, counter, renderer, gpu_lock,
               stop_event, prof, reset_flag, t_start, audit=None, debug=False,
               publish_api=False, force_focus=False, record_dir=None,
               record_desired=False, det_overrides=None, stats=None,
               hot_reloader=None):
    """Inference loop: consumes every raw frame the display thread delivers
    (acquire rate, no idle throttling — ByteTrack needs temporal density for
    accurate background counting), runs the model, and stores the overlay.

    MJPEG + render are skipped when this node is not focused (Lazy Rendering);
    detection, tracking and counting never are. `force_focus` pins a node to
    full rendering regardless of the API focus state (single-node mode).

    Recording is per-node: `record_desired` seeds the initial state (--record),
    and the `record` command toggles it. While desired, a per-node writer
    opened at the raw resolution gets every annotated frame, with or without
    UI subscribers. The AI loop runs at acquire rate, so the file paces to
    the source fps."""
    last_seq = 0
    current_day = _stats_mod.today_wib()
    start_time = time.time()
    peak_occupancy = 0
    saved_node = stats.node(node_id) if stats else None
    hourly_counts = dict(saved_node["hourly_counts"]) if saved_node else {}
    slot_counts = dict(saved_node["slot_counts"]) if saved_node else {}
    session_starts = list(saved_node["session_starts"]) if saved_node else []
    start_hhmm = _stats_mod.hhmm_wib()
    if start_hhmm not in session_starts:
        session_starts.append(start_hhmm)
    if stats is not None:
        stats.record(
            node_id, counter, hourly_counts,
            slot_counts=slot_counts, session_starts=session_starts,
        )
    writer = None          # per-node cv2.VideoWriter, open while recording
    record_desired = bool(record_desired)
    if record_desired:
        log.info("Node %s starts recording (--record)", node_id)
    while not stop_event.is_set():
        # Blocking: take the next raw frame the display thread delivers. The
        # display runs at the source's native rate, so this paces the AI loop
        # at acquire rate; no extra sleep/throttle in between.
        t = time.perf_counter()
        raw, last_seq = bus.wait_raw(last_seq, stop_event, timeout=1.0)
        t_acquire = time.perf_counter() - t

        # Midnight (WIB): close yesterday's snapshot once, and reset EVERY node
        # locally whenever today_wib() changes. Checked before the frame guard
        # so a node whose stream is down still rolls over on time.
        now_day = _stats_mod.today_wib()
        roll = stats.rollover() if stats is not None else None
        if roll or now_day != current_day:
            current_day = now_day
            counter.reset()
            hourly_counts.clear()
            slot_counts.clear()
            session_starts[:] = [_stats_mod.hhmm_wib()]
            peak_occupancy = 0
            start_time = time.time()
            if publish_api:
                seed_node_data(node_id, {
                    "motor": 0,
                    "mobil": 0,
                    "truk": 0,
                    "hourly_counts": {},
                    "slot_counts": {},
                    "session_starts": list(session_starts),
                })
            if roll:
                log.info("Midnight rollover completed for %s -> %s",
                         roll["from"], roll["to"])
            else:
                log.info("Midnight rollover reset node %s -> %s",
                         node_id, current_day)

        if hot_reloader is not None:
            hot_reloader.check_and_reload(target_node_id=node_id)

        if raw is None:
            # No frame reached the AI loop (stalled HLS/RTSP / reconnecting):
            # surface it so the dashboard can show RECONNECTING immediately.
            mark_node_reconnecting(node_id)
            continue
        frame, arrival, wall = raw
        set_node_frame_size(node_id, frame.shape[1], frame.shape[0])
        focused = force_focus or get_active_focus() == node_id

        command = pop_command(node_id)
        # update_zone arrives as {"action": "update_zone", "points": [[0-1], ...]}.
        # The polygon is rebuilt against the CURRENT raw frame so the ratios
        # land on the right pixels whatever the stream resolution is; on
        # failure the zone config from settings.yaml stays active.
        if isinstance(command, dict) and command.get("action") == "update_zone":
            if not command.get("applied"):
                try:
                    frame_h, frame_w = frame.shape[:2]
                    pixel_pts = [[int(x * frame_w), int(y * frame_h)]
                                 for x, y in command["points"]]
                    if hot_reloader is not None:
                        hot_reloader.apply_node_zones(
                            node_id,
                            command["points"],
                            frame_width=frame_w,
                            frame_height=frame_h,
                        )
                    else:
                        counter.update_polygon(
                            command["points"], frame_w, frame_h,
                            preserve_zone_attrs=True,
                        )
                        if renderer is not None:
                            renderer.update_zones(counter.zones)
                        if det_overrides is not None:
                            det_overrides["polygons"] = list(counter.zones)
                        det.configure_perspective_from_polygons(counter.zones)
                    persist_zone(
                        node_id,
                        pixel_pts,
                        path=hot_reloader.nodes_path if hot_reloader is not None else None,
                    )
                    log.info("Node %s zone updated + saved: %d pts on frame %dx%d",
                             node_id, len(command["points"]), frame_w, frame_h)
                except (KeyError, TypeError, ValueError) as e:
                    log.warning("Node %s rejected update_zone (%s)", node_id, e)
            command = None
        if command == "record" or reset_flag.get("record"):
            reset_flag["record"] = False
            record_desired = not record_desired
            log.info("Node %s record -> %s",
                     node_id, "ON" if record_desired else "OFF")

        # Writer follows the desired state (open lazily, close on toggle-off).
        if record_desired and writer is None:
            writer = _open_writer(frame, node_id, record_dir,
                                  streamer.effective_fps())
        elif not record_desired and writer is not None:
            writer.release()
            writer = None
            log.info("Node %s recording saved", node_id)

        if reset_flag["v"] or command == "reset":
            counter.reset()
            reset_flag["v"] = False
            clear_commands(node_id)
            start_time = time.time()
            peak_occupancy = 0
            hourly_counts.clear()
            slot_counts.clear()
            session_starts[:] = [_stats_mod.hhmm_wib()]
            log.info("Node %s counts reset", node_id)

        try:
            # ---- [RT-DETR + ByteTrack INFERENCE] ----
            # detect() runs on the shared engine; gpu_lock serializes calls so
            # one TensorRT context serves all nodes. t_infer includes the wait
            # for that lock, which is the GPU-contention signal.
            t = time.perf_counter()
            with gpu_lock:
                raw_dets = det.detect(frame, **(det_overrides or {}))
            t_infer = time.perf_counter() - t
            t = time.perf_counter()
            dets = counter.track(raw_dets, frame)
            t_track = time.perf_counter() - t
            events = counter.update(dets, now=arrival,
                                    freeze=streamer.is_frozen())
        except Exception as e:
            log.warning("node %s detector: %s", node_id, e)
            continue

        current_occupancy = len(dets)
        peak_occupancy = max(peak_occupancy, current_occupancy)
        fold_slot_counts(slot_counts, hourly_counts, events)
        if stats is not None:
            # Throttled: the in-memory snapshot follows every count, the file
            # at most once per save_interval_sec.
            stats.record(
                node_id, counter, hourly_counts,
                slot_counts=slot_counts, session_starts=session_starts,
            )
            stats.maybe_save()

        if audit:
            for ev in events:
                audit.write(ev)
        if debug and events:
            for ev in events:
                old = ev.get("reattributed_from")
                if old:
                    log.info("REATTR %s -> %s id=%d zone=%d (total=%d)",
                             old, ev["class_name"], ev["track_id"], ev["zone"],
                             counter.total)
                else:
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

        # Recording: annotate on a copy and write, subscriber or not. The AI
        # loop runs at acquire rate, so the file paces to the source fps.
        if writer is not None:
            out = frame.copy()
            if renderer is not None:
                renderer.draw(out, snap, dict(counter.total_by_class),
                              trails={k: list(v)
                                      for k, v in counter.trails.items()})
            writer.write(np.ascontiguousarray(out))

        if publish_api:
            totals = counter.total_by_class
            update_node_data(node_id, {
                "motor": totals.get("motor", 0),
                "mobil": totals.get("mobil", 0),
                "truk": totals.get("truk/bus", 0),
                "occupancy": current_occupancy,
                "fps": counter.n_frames / max(time.monotonic() - t_start, 1e-6),
                "start_time": start_time,
                "peak_occupancy": peak_occupancy,
                "hourly_counts": {h: dict(b) for h, b in hourly_counts.items()},
                "slot_counts": {s: dict(b) for s, b in slot_counts.items()},
                "session_starts": list(session_starts),
            })

        prof.tick_ai(len(dets), infer=t_infer, track=t_track, acquire=t_acquire)
        prof.maybe_log()

    if writer is not None:
        writer.release()
        log.info("Node %s recording saved on shutdown", node_id)


def _open_writer(frame, node_id, record_dir, fps):
    """Per-node annotated-video sink: logs/output_{node_id}_{ts}.mp4.
    mp4v first, AVI/XVID fallback for Windows builds without an MP4 backend.
    fps follows the AI loop = the acquire rate, so playback is real-time."""
    h, w = frame.shape[:2]
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(record_dir) if record_dir else Path("logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    for fourcc, ext in (("mp4v", "mp4"), ("XVID", "avi")):
        path = out_dir / f"output_{node_id}_{ts}.{ext}"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc),
                             round(fps) or 25, (w, h))
        if vw.isOpened():
            log.info("Node %s audit video: %s", node_id, path)
            return vw
        vw.release()
        Path(path).unlink(missing_ok=True)
    log.warning("no working VideoWriter codec; recording disabled for %s",
                node_id)
    return None


def _display_worker(node_id, streamer, bus, renderer, stop_event, prof, sink,
                    focused=None, deadline=None):
    """Fast loop. Sole consumer of the stream queue: pulls every frame at the
    source's native rate, overlays the newest AI result, and hands the frame
    to `sink` (imshow or MJPEG publish). Never blocks on inference.

    `focused(node_id)` gates the expensive path (render + JPEG encode): an
    unwatched node still publishes raw frames to the AI but skips drawing and
    streaming entirely. Defaults to always-focused (single-node window).

    Pacing matters because FFmpeg's HLS demuxer buffers whole segments and
    releases them in a burst (~0ms gaps, then a multi-second stall). Draining
    that burst unpaced looks like fast-forward, so each frame is spaced to the
    source rate and the bounded queue backpressures the pump into real time."""
    next_frame = time.monotonic()
    if focused is None:
        focused = lambda _nid: True
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
        # Lazy Rendering: boxes + JPEG encode only for the watched node.
        # Detection/counting in the AI worker is unaffected.
        if focused(node_id):
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
        else:
            t_render = t_publish = 0.0

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
    zone_cfgs = cfg.get("zones", [])
    zones = build_zones(zone_cfgs)
    audit = AuditLogger("logs/count_audit.csv") if debug else None

    streamer, counter, renderer = _build_pipeline(
        cfg, zones, _expand_source(sc["source"]), debug
    )
    det = _build_detector(cfg, polygons=zones)
    editor = ZoneEditor(zones, zone_cfgs)
    stats = StatsStore.from_config(cfg.get("persistence"))
    stats.load()
    register_stats_store(stats)
    _seed_api_from_stats(stats)
    seed_counter(counter, stats.node(node_id))

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    gpu_lock = threading.Lock()
    reloader = ConfigHotReloader(
        settings_path=config_path or "config/settings.yaml",
        nodes_path="config/nodes.yaml",
        det=det,
        gpu_lock=gpu_lock,
    )
    reloader.register_node(node_id, counter, renderer)
    global _ACTIVE_RELOADER
    with _ACTIVE_RELOADER_LOCK:
        _ACTIVE_RELOADER = reloader
    register_zone_update_hook(apply_node_zones)
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

    if show:
        cv2.namedWindow("ATCS", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("ATCS", editor.on_mouse)

    def sink(out):
        # Runs on the main thread: imshow/highgui stays on one thread.
        if not show:
            return
        cv2.imshow("ATCS", out)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            stop.set()
        elif k in (ord("c"), ord("C")):
            reset_flag["record"] = True
        elif k in (ord("r"), ord("R")):
            # Defer to the AI thread; it owns the counter.
            reset_flag["v"] = True
            log.info("Counts reset")
        elif k in (ord("s"), ord("S")) and editor.dirty and config_path:
            save_zones(config_path, zones, zone_cfgs)
            reloader.apply_node_zones(node_id, zones)
            zones[:] = counter.zones
            editor.zones = counter.zones
            editor.dirty = False
            log.info("Zones saved")

    t0 = time.monotonic()
    ai = threading.Thread(
        target=_ai_worker,
        args=(node_id, bus, streamer, det, counter, renderer, gpu_lock,
              stop, prof, reset_flag, t0, audit, debug, serve_api, True,
              None, bool(record)),
        kwargs={"stats": stats, "hot_reloader": reloader},
        daemon=True, name="ai")
    ai.start()

    deadline = t0 + duration if duration else None
    try:
        _display_worker(node_id, streamer, bus, renderer, stop, prof, sink,
                        focused=lambda _nid: True, deadline=deadline)
    finally:
        register_zone_update_hook(None)
        register_stats_store(None)
        with _ACTIVE_RELOADER_LOCK:
            if _ACTIVE_RELOADER is reloader:
                _ACTIVE_RELOADER = None
        stop.set()
        ai.join(timeout=5.0)
        streamer.stop()
        if audit:
            audit.close()
        stats.save()   # final flush: the throttled writer may be behind
        cv2.destroyAllWindows()
        elapsed = time.monotonic() - t0
        log.info("Final: total=%d %s", counter.total, dict(counter.total_by_class))
        log.info("Processed %d of %d delivered frames in %.0fs (%.1f fps), "
                 "dedup-suppressed=%d, connects=%d",
                 counter.n_frames, streamer.frames_delivered, elapsed,
                 counter.n_frames / max(elapsed, 1e-6), counter.suppressed,
                 streamer.connects)
    return 0


def run_multi(cfg, nodes, duration=None, debug=False,
              config_path="config/settings.yaml", nodes_path="config/nodes.yaml"):
    """Async pipeline per CCTV node, headless.

    Shared: a single Detector behind gpu_lock (one TensorRT engine in memory,
    serialized GPU inference). Per node: VideoStreamer + AI thread + display
    thread. The AI thread pushes counts via update_node_data; the display
    thread encodes the annotated frame via set_latest_frame; the frontend
    picks a node to watch and /set-focus decides which one runs full rate.
    """
    sc = cfg["stream"]
    zone_cfgs = cfg.get("zones", [])

    det = _build_detector(cfg, polygons=zone_cfgs)
    gpu_lock = threading.Lock()
    reloader = ConfigHotReloader(
        settings_path=config_path,
        nodes_path=nodes_path,
        det=det,
        gpu_lock=gpu_lock,
    )
    global _ACTIVE_RELOADER
    with _ACTIVE_RELOADER_LOCK:
        _ACTIVE_RELOADER = reloader
    register_zone_update_hook(apply_node_zones, nodes_path=nodes_path)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    t_run = time.monotonic()
    stats = StatsStore.from_config(cfg.get("persistence"))
    stats.load()
    register_stats_store(stats)
    _seed_api_from_stats(stats)

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
        source = _expand_source(n.get("source") or sc["source"])
        zones = build_zones(n.get("zones") or zone_cfgs)
        # Optional per-node detector tuning from nodes.yaml (camera tilt /
        # distance differ per intersection); settings.yaml is the fallback.
        det_over = {
            k: (bool(n[k]) if k == "perspective_scaling" else float(n[k]))
            for k in ("truck_min_area", "car_min_area", "truck_min_height_ratio",
                      "perspective_scaling", "y_ref_far", "y_ref_mid", "y_ref_near",
                      "scale_far", "scale_near")
            if k in n and n[k] is not None
        }
        # Per-node counter tuning (min_displacement_px, car lock, flow gate)
        # overrides settings.yaml; unknown node keys are ignored.
        counter_keys = set(cfg.get("counter", {})) | {
            "max_stale_frames", "allowed_directions", "flow_max_angle_deg",
            "truck_min_area", "truck_confirm_frames", "truck_min_height_ratio",
            "min_track_frames",
        }
        counter_cfg = {**cfg.get("counter", {}),
                       **{k: n[k] for k in counter_keys if k in n}}
        node_cfg = {**cfg, "detector": {**cfg["detector"], **det_over}, "counter": counter_cfg}
        streamer, counter, renderer = _build_pipeline(
            node_cfg, zones, source, debug)
        det.configure_perspective_from_polygons(zones)
        det_over["polygons"] = zones
        reloader.register_node(node_id, counter, renderer, det_over)
        seed_counter(counter, stats.node(node_id))
        streamer.start()
        bus = FrameBus()
        prof = StageProfiler("node:" + node_id)
        reset_flag = {"v": False}
        t0 = time.monotonic()

        ai = threading.Thread(
            target=_ai_worker,
            args=(node_id, bus, streamer, det, counter, renderer, gpu_lock,
                  stop, prof, reset_flag, t0, None, debug, True, False,
                  "logs", False, det_over),
            kwargs={"stats": stats, "hot_reloader": reloader},
            daemon=True, name=f"ai-{node_id}")
        disp = threading.Thread(
            target=_display_worker,
            args=(node_id, streamer, bus, renderer, stop, prof,
                  lambda out, nid=node_id: set_latest_frame(nid, out)),
            kwargs={"focused": lambda _nid: get_active_focus() == _nid},
            daemon=True, name=f"disp-{node_id}")
        ai.start()
        disp.start()
        workers.append((node_id, streamer, ai, disp, counter))
        log.info("node %s: streaming %s", node_id, redact_url(source))

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
    register_zone_update_hook(None)
    register_stats_store(None)
    with _ACTIVE_RELOADER_LOCK:
        if _ACTIVE_RELOADER is reloader:
            _ACTIVE_RELOADER = None
    stats.save()   # final flush: the throttled writer may be behind
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
    a.add_argument("--clear-stats", action="store_true",
                   help="reset logs/daily_stats.json for today and exit (or start fresh)")
    a.add_argument("--prune-min-hourly", type=int, default=None,
                   help="remove hourly buckets below <int> total vehicles and save immediately")
    a.add_argument("--min-uptime-minutes", type=float, default=None,
                   help="override minimum session uptime in minutes before persisting stats (default: 15.0)")
    args = a.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    setup_logging(level=cfg.get("logging", {}).get("level", "INFO"),
                  log_file=cfg.get("logging", {}).get("file", "logs/app.log"))
    if args.min_uptime_minutes is not None:
        cfg.setdefault("persistence", {})["min_persist_uptime_seconds"] = max(
            0.0, float(args.min_uptime_minutes) * 60.0
        )
    if args.clear_stats or args.prune_min_hourly is not None:
        stats = StatsStore.from_config(cfg.get("persistence"))
        if args.clear_stats:
            stats.clear(save_to_disk=True)
            log.info("Cleared daily stats in %s", stats.path)
        else:
            stats.load()
        if args.prune_min_hourly is not None:
            pruned = stats.prune_hourly_below(args.prune_min_hourly)
            stats.save(force=True)
            log.info("Pruned %d hourly bucket(s) below %d vehicles in %s",
                     pruned, args.prune_min_hourly, stats.path)
        if not (args.multi or args.api or args.duration or args.record or args.no_display):
            sys.exit(0)
    if args.multi:
        from src.api.server import load_node_registry
        reg = load_node_registry()
        if args.nodes:
            keep = {n.strip() for n in args.nodes.split(",")}
            reg = [n for n in reg if n["id"] in keep]
        if not reg:
            raise SystemExit("no nodes selected (check config/nodes.yaml / --nodes)")
        sys.exit(run_multi(cfg, reg, duration=args.duration, debug=args.debug,
                           config_path=args.config))
    sys.exit(run(cfg, show=not args.no_display, config_path=args.config,
                 duration=args.duration, debug=args.debug, record=args.record,
                 serve_api=args.api))
