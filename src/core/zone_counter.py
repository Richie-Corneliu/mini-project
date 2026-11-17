"""C-BIoU tracking + polygon-zone crossing counting.

A track counts in a zone when its centroid is inside the polygon, or when the
segment prev_centroid -> centroid crosses a zone edge: HLS bursts let fast
motorcycles jump over a thin polygon between frames. The permanent per-zone
id latch stops recounts; spatial-temporal dedup (dedup_radius_px +
dedup_cooldown_sec) only catches one-frame re-ID ghosts — an id seen
STABLE_FRAMES+ times with conf >= STABLE_CONF is trusted over the spatial
check, so swarm riders passing the same spot count individually.
Class uses mode of the last CLASS_WINDOW frames, conf breaks ties.

Three class-accuracy filters run before a count is accepted:
  1. Displacement gate: a track must move >= min_displacement_px from its first
     seen centroid, else it is a static false positive (bush in the wind).
  2. Majority vote: the counted class is the mode of the track's class history,
     not the instantaneous frame prediction (fixes car/truck flicker).
  3. Cabin suppression: a car voted inside a truck/bus box is the truck cabin,
     not a vehicle, and is discarded.

Post-count class drift is handled by atomic re-attribution: a counted id that
stays inside the zone gets its class re-voted every frame, and a stabilized
flip (truck -> car mid-zone on a pickup) shifts total_by_class without ever
touching the total. Motorcycle inertia: once a track has >= motor_lock_frames
motorcycle observations, a car run shorter than motor_flip_frames is treated
as a turning bike's RT-DETR flicker and cannot flip the resolved class.
Car inertia mirrors it against truck/bus inflation: a track seen as car
car_lock_frames times (and not opening as truck/bus in its first 2 frames)
holds car through a truck/bus run of <= car_flip_frames — approaching
perspective growth and diagonal turns inflate the box, not the vehicle — and
an already-counted car id is never re-attributed to truck/bus.

Two RTSP-specific guards sit around those filters:
  * Pre-zone anchor — the displacement gate's origin (_start) is recorded on
    the first frame a tracker_id appears ANYWHERE in the view, not on zone
    entry, so a fast vehicle tracked for a frame or two outside the polygon
    still clears min_displacement_px on the single frame it lands inside,
    while a jittering roadside object never drifts from its birth centroid.
  * Ghost handover — an id counted inside a zone leaves a ghost (its last
    in-zone centroid plus the frame it was last seen). A new id born strictly
    after that frame (so the two never shared a frame), within
    ghost_ttl_frames of it and born within ghost_radius_px of it, inherits the
    counted latch and the class history instead of counting the same vehicle
    twice when ByteTrack drops the id under occlusion or stream stutter.
"""
from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import supervision as sv

from .detector import centroids


_DIRECTION_VECTORS: Dict[str, Tuple[float, float]] = {
    "down": (0.0, 1.0),
    "up": (0.0, -1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "down-left": (-1.0, 1.0),
    "down-right": (1.0, 1.0),
    "up-left": (-1.0, -1.0),
    "up-right": (1.0, -1.0),
}


def _normalize_directions(specs) -> List[Tuple[float, float]]:
    """Convert direction tokens or 2-element [dx, dy] vectors into unit tuples."""
    if not specs:
        return []
    out: List[Tuple[float, float]] = []
    for spec in specs:
        if isinstance(spec, str):
            key = spec.strip().lower().replace("_", "-")
            vec = _DIRECTION_VECTORS.get(key)
            if vec is None:
                raise ValueError(f"unknown direction spec: {spec!r}")
            dx, dy = vec
        else:
            dx, dy = float(spec[0]), float(spec[1])
        norm = math.hypot(dx, dy)
        if norm <= 1e-9:
            continue
        out.append((dx / norm, dy / norm))
    return out


def _orient(a, b, c) -> int:
    v = (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1])) \
        - (float(b[1]) - float(a[1])) * (float(c[0]) - float(a[0]))
    return (v > 0) - (v < 0)


def segments_cross(a, b, c, d) -> bool:
    """True if segment a-b crosses segment c-d (proper crossing only)."""
    if max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0]) \
            or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1]):
        return False
    return (_orient(a, b, c) != _orient(a, b, d)
            and _orient(c, d, a) != _orient(c, d, b))


def _crosses_polygon(poly: np.ndarray, a, b) -> bool:
    """True if segment a-b crosses any edge of the polygon."""
    n = len(poly)
    for k in range(n):
        if segments_cross(a, b, poly[k], poly[(k + 1) % n]):
            return True
    return False


class ZoneCounter:
    CLASS_WINDOW = 15    # frames of class history for the crossing vote
    TRAIL_LEN = 30       # debug trajectory tail length
    STABLE_FRAMES = 5    # frames in view before a track id beats spatial dedup
    STABLE_CONF = 0.35   # current-frame conf floor for that trust
    CAR_CLS = 2          # COCO id for car
    MOTOR_CLS = 3        # COCO id for motorcycle
    TRUCK_CLS = (5, 7)   # COCO bus + truck
    MOTOR_LOCK_FRAMES = 3   # motor sightings that lock a track's class
    MOTOR_FLIP_FRAMES = 4   # car run <= this stays a flicker under the lock
    CAR_LOCK_FRAMES = 3     # car sightings that lock a track's class
    CAR_FLIP_FRAMES = 6     # truck run <= this stays a flicker under the lock
    GHOST_TTL_FRAMES = 6    # frames a counted id stays claimable after vanishing
    GHOST_RADIUS_PX = 35.0  # max birth-centroid distance for a ghost handover
    MAX_STALE_FRAMES = 300  # frames before inactive tracker state is pruned
    FLOW_MAX_ANGLE_DEG = 75.0  # max angular deviation from allowed flow direction
    MIN_DISPLACEMENT_PX = 12.0
    MIN_TRACK_FRAMES = 1
    ZONE_DWELL_FRAMES = 1
    TRUCK_MIN_AREA = 12000.0
    TRUCK_PROMOTE_MAX_ASPECT = 2.0

    def __init__(self, zones: List[Union[sv.PolygonZone, dict]],
                 class_labels: Optional[Dict[int, str]] = None,
                 tracker_config: Optional[dict] = None,
                 dedup_cooldown_sec: float = 4.0, dedup_radius_px: float = 10.0,
                 min_displacement_px: float = MIN_DISPLACEMENT_PX,
                 class_window: int = CLASS_WINDOW,
                 motor_lock_frames: int = MOTOR_LOCK_FRAMES,
                 motor_flip_frames: int = MOTOR_FLIP_FRAMES,
                 car_lock_frames: int = CAR_LOCK_FRAMES,
                 car_flip_frames: int = CAR_FLIP_FRAMES,
                 cabin_iou: float = 0.6, cabin_area_ratio: float = 0.55,
                 still_count_frames: int = 30, still_min_conf: float = 0.35,
                 ghost_ttl_frames: int = GHOST_TTL_FRAMES,
                 ghost_radius_px: float = GHOST_RADIUS_PX,
                 max_stale_frames: int = MAX_STALE_FRAMES,
                 allowed_directions: Optional[List[Union[str, List[float], Tuple[float, float]]]] = None,
                 flow_max_angle_deg: float = FLOW_MAX_ANGLE_DEG,
                 min_track_frames: int = MIN_TRACK_FRAMES,
                 zone_dwell_frames: int = ZONE_DWELL_FRAMES,
                 truck_min_area: float = TRUCK_MIN_AREA,
                 truck_promote_area: Optional[float] = None,
                 truck_promote_max_aspect: float = TRUCK_PROMOTE_MAX_ASPECT,
                 debug: bool = False):
        from trackers import CBIoUTracker

        self.zones: List[sv.PolygonZone] = []
        self._zone_lock = threading.RLock()
        self._zone_allowed_directions = []
        self._zone_flow_max_angle_deg = []
        for z in zones:
            if isinstance(z, dict):
                pz = sv.PolygonZone(polygon=np.array(z["polygon"], dtype=np.float32))
                ad = z.get("allowed_directions")
                ang = z.get("flow_max_angle_deg")
                if ad is not None:
                    pz.allowed_directions = ad
                if ang is not None:
                    pz.flow_max_angle_deg = ang
                self.zones.append(pz)
                self._zone_allowed_directions.append(ad)
                self._zone_flow_max_angle_deg.append(ang)
            else:
                self.zones.append(z)
                self._zone_allowed_directions.append(getattr(z, "allowed_directions", None))
                self._zone_flow_max_angle_deg.append(getattr(z, "flow_max_angle_deg", None))
        self.class_labels = class_labels or {}
        self.dedup_cooldown = float(dedup_cooldown_sec)
        self.dedup_r2 = float(dedup_radius_px) ** 2
        self.min_disp2 = float(min_displacement_px) ** 2
        self.class_window = int(class_window)
        self.motor_lock_frames = int(motor_lock_frames)
        self.motor_flip_frames = int(motor_flip_frames)
        self.car_lock_frames = int(car_lock_frames)
        self.car_flip_frames = int(car_flip_frames)
        self.truck_labels = {self._label(c) for c in self.TRUCK_CLS}
        self.cabin_iou = float(cabin_iou)
        self.cabin_area_ratio = float(cabin_area_ratio)
        self.still_count_frames = int(still_count_frames)
        self.still_min_conf = float(still_min_conf)
        self.ghost_ttl_frames = int(ghost_ttl_frames)
        self.ghost_r2 = float(ghost_radius_px) ** 2
        self.max_stale_frames = max(int(max_stale_frames), self.ghost_ttl_frames + 1)
        self.allowed_directions = allowed_directions
        self.flow_max_angle_deg = float(flow_max_angle_deg)
        self.min_track_frames = max(1, int(min_track_frames))
        self.zone_dwell_frames = max(1, int(zone_dwell_frames))
        self.truck_min_area = float(truck_min_area)
        self._explicit_truck_promote_area = truck_promote_area is not None
        self.truck_promote_area = (
            self.truck_min_area * 1.25 if truck_promote_area is None
            else float(truck_promote_area)
        )
        self.truck_promote_max_aspect = float(truck_promote_max_aspect)
        self._zone_y_mid = self._compute_zone_y_mid()
        _normalize_directions(self.allowed_directions)
        for zd in self._zone_allowed_directions:
            _normalize_directions(zd)
        self.debug = bool(debug)
        self.tracker = CBIoUTracker(**(tracker_config or {
            "lost_track_buffer": 90,
            "track_activation_threshold": 0.20,
            "minimum_consecutive_frames": 1,
        }))
        self.total = 0
        self.suppressed = 0
        self.suppressed_cabin = 0
        self.suppressed_direction = 0
        self.total_by_class = defaultdict(int)
        self.n_frames = 0
        self.class_hist = {}   # tid -> deque of (class_id, confidence)
        # counted_id_class[zi][tid] = label registered when this id first
        # counted in zone zi. Drives atomic re-attribution: a counted id whose
        # stabilized class drifts while still active in the zone moves its
        # count from the old class bucket to the new one (total unchanged).
        self.counted_id_class = [dict() for _ in self.zones]
        self.trails = {}       # tid -> deque of centroid pts (debug render)
        self._first_cls = {}   # tid -> first 2 class ids, for the car lock
        self._car_lock_latched = set()  # tids permanently locked against truck/bus upgrade
        self._truck_lock_latched = set()  # tids locked as truk/bus (receding or promoted)
        self._truck_frames = {}         # tid -> total TRUCK_CLS observations
        self._consec_valid_truck = {}   # tid -> consecutive perspective-validated TRUCK_CLS frames
        self._peak_truck_area = {}      # tid -> max bbox area seen under TRUCK_CLS
        self._last_area = {}            # tid -> most recent bbox area
        self._last_truck_cls = {}       # tid -> most recent COCO truck/bus class id (5 or 7)
        self._prev = {}        # tid -> last centroid, for jump-over-edge sweep
        self._start = {}       # tid -> first centroid, for the displacement gate
        self._age = {}         # tid -> frames seen, for the stable-id trust rule
        self._missing = {}     # tid -> frames since last seen (hist eviction)
        self._zone_still = {}  # tid -> {zi: frames blocked by displacement gate}
        self._counted = [set() for _ in self.zones]   # permanent per-zone id latch
        self._recent = [deque() for _ in self.zones]  # (t, x, y) accepted count events
        # ---- ghost handover bookkeeping --------------------------------
        # A counted id that vanishes inside a zone leaves a "ghost": its last
        # in-zone centroid plus the frame it was last seen. A tracker_id born
        # AFTER that (never coexisting) within ghost_ttl_frames and born within
        # ghost_radius_px of the ghost inherits the counted latch instead of
        # counting the same vehicle twice (occlusion / RTSP micro-stutter).
        self._frame_idx = 0            # monotonic frame clock, gaps included
        self._born_frame = {}          # tid -> _frame_idx first seen anywhere
        self._birth_centroid = {}      # tid -> centroid on that first frame
        self._last_seen_frame = {}     # tid -> _frame_idx last seen
        self._last_centroid = {}       # tid -> last known centroid
        self._zone_ghost = [dict() for _ in self.zones]    # zi -> {old_tid: centroid}
        self.ghost_handovers = 0
        # no separate _last_centroid: self._prev already is the last centroid,
        # and a ghost keeps the last IN-ZONE one in _zone_ghost.

    def track(self, dets: sv.Detections, frame: np.ndarray) -> sv.Detections:
        return self.tracker.update(dets)

    def _label(self, cls: int) -> str:
        return self.class_labels.get(cls, f"class_{cls}")

    def _voted_class(self, tid: int, fallback: int) -> int:
        """Majority class over the track's recent history. On a tie the class
        with the higher summed detection confidence wins."""
        h = self.class_hist.get(tid)
        if not h:
            return fallback
        votes = defaultdict(lambda: [0, 0.0])
        for cls, conf in h:
            votes[cls][0] += 1
            votes[cls][1] += conf
        return max(votes, key=lambda c: (votes[c][0], votes[c][1]))

    def _resolved_class(self, tid: int, fallback: int) -> int:
        """Vote with motorcycle/car turn inertia applied.

        Motor: once a track has >= motor_lock_frames motorcycle sightings, a
        non-motor run of <= motor_flip_frames frames is a turning bike's
        RT-DETR flicker and the resolved class stays motorcycle. The run is
        the longest consecutive non-motor stretch in the history (not the
        trailing run — the crossing frame itself often flickers back to
        motorcycle, which would otherwise mask an 8-frame real car run). A
        longer non-motor run means the vote wins: a genuine transition
        (pickup entering as truck, later reading car) re-attributes.

        Car: same shape against truck/bus. A track that opened as car and has
        car_lock_frames car sightings cannot upgrade to truck/bus on a run of
        <= car_flip_frames — approaching the camera and diagonal turns inflate
        the axis-aligned box, not the vehicle. A track whose first 2 frames
        read truck/bus is a real truck and keeps the truck -> car path open.
        """
        h = self.class_hist.get(tid)
        if not h:
            return fallback
        vote = self._voted_class(tid, fallback)
        if vote == self.MOTOR_CLS:
            return vote
        motor_seen = sum(1 for cls, _ in h if cls == self.MOTOR_CLS)
        if (motor_seen >= self.motor_lock_frames
                and self._longest_non_run(h, self.MOTOR_CLS) <= self.motor_flip_frames):
            return self.MOTOR_CLS
        if tid in self._truck_lock_latched:
            start = self._start.get(tid)
            last_pt = self._prev.get(tid)
            receding = (start is not None and last_pt is not None
                        and last_pt[1] < start[1])
            peak_a = self._peak_truck_area.get(tid, 0.0)
            curr_a = self._last_area.get(tid, 0.0)
            shrinking = peak_a > 0.0 and curr_a < 0.85 * peak_a
            consec_car = 0
            for c, _ in reversed(h):
                if c == self.CAR_CLS:
                    consec_car += 1
                else:
                    break
            if (receding or shrinking or consec_car < 3
                    or peak_a >= self.truck_promote_area):
                return self._last_truck_cls.get(tid, 7)
        fc = self._first_cls.get(tid, ())
        if vote in self.TRUCK_CLS and (
            self._car_locked(tid)
            or (fc and fc[0] == self.CAR_CLS and tid not in self._truck_lock_latched)
        ):
            return self.CAR_CLS
        return vote

    @staticmethod
    def _longest_non_run(hist, cls: int) -> int:
        """Longest consecutive stretch of classes other than `cls`."""
        longest = run = 0
        for c, _ in hist:
            run = 0 if c == cls else run + 1
            longest = max(longest, run)
        return longest

    def _car_locked(self, tid: int) -> bool:
        """True once a track opened as car (after >=2 sightings) or accumulated
        car_lock_frames car sightings without opening as truck/bus or promoting."""
        if tid in self._truck_lock_latched:
            return False
        if tid in self._car_lock_latched:
            return True
        if any(c in self.TRUCK_CLS for c in self._first_cls.get(tid, ())):
            return False
        h = self.class_hist.get(tid)
        return bool(h) and sum(1 for c, _ in h if c == self.CAR_CLS) >= self.car_lock_frames

    def _compute_zone_y_mid(self) -> Optional[float]:
        ys = [
            float(p[1])
            for z in self.zones
            for p in np.asarray(z.polygon, dtype=float).reshape(-1, 2)
        ]
        return 0.5 * (min(ys) + max(ys)) if ys else None

    def update_thresholds(
        self,
        *,
        min_displacement_px: Optional[float] = None,
        allowed_directions=...,
        flow_max_angle_deg: Optional[float] = None,
        ghost_ttl_frames: Optional[int] = None,
        ghost_radius_px: Optional[float] = None,
        truck_min_area: Optional[float] = None,
    ) -> None:
        """Apply hot-reloaded YAML thresholds in place without dropping latches."""
        if min_displacement_px is not None:
            self.min_disp2 = float(min_displacement_px) ** 2
        if allowed_directions is not ...:
            _normalize_directions(allowed_directions)
            self.allowed_directions = allowed_directions
        if flow_max_angle_deg is not None:
            self.flow_max_angle_deg = float(flow_max_angle_deg)
        if ghost_ttl_frames is not None:
            self.ghost_ttl_frames = int(ghost_ttl_frames)
            self.max_stale_frames = max(self.max_stale_frames, self.ghost_ttl_frames + 1)
        if ghost_radius_px is not None:
            self.ghost_r2 = float(ghost_radius_px) ** 2
        if truck_min_area is not None:
            self.truck_min_area = float(truck_min_area)
            if not self._explicit_truck_promote_area:
                self.truck_promote_area = self.truck_min_area * 1.25

    def _motor_locked(self, tid: int) -> bool:
        """Read-only mirror of the motor-inertia branch in `_resolved_class`."""
        h = self.class_hist.get(tid)
        if not h:
            return False
        return (sum(1 for c, _ in h if c == self.MOTOR_CLS) >= self.motor_lock_frames
                and self._longest_non_run(h, self.MOTOR_CLS) <= self.motor_flip_frames)

    def get_track_diagnostics(self, tid: int,
                              bbox: Optional[Tuple] = None) -> dict:
        """Per-track telemetry for the diagnostic HUD (`ui.show_diagnostics`).

        Pure read: touches no counting state, so it is safe to call from the
        display thread while the AI thread runs `update()`. `bbox` is the live
        (x1, y1, x2, y2) row; without it `area_px` is None.
        """
        area = None
        if bbox is not None:
            b = np.asarray(bbox, dtype=float).ravel()
            if b.size == 4:
                area = int(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))
        start, last = self._start.get(tid), self._prev.get(tid)
        disp = None
        if start is not None and last is not None:
            disp = round(float(((last[0] - start[0]) ** 2
                                + (last[1] - start[1]) ** 2) ** 0.5), 1)
        return {"tid": int(tid),
                "area_px": area,
                "displacement_px": disp,
                "car_locked": self._car_locked(tid),
                "motor_locked": self._motor_locked(tid),
                "counted_zones": [zi for zi in range(len(self.zones))
                                  if tid in self._counted[zi]]}

    def diagnostics_summary(self) -> dict:
        """Counter health for the diagnostic strip. Pure read as well."""
        return {"handovers": self.ghost_handovers,
                "suppressed": self.suppressed,
                "suppressed_cabin": self.suppressed_cabin,
                "suppressed_direction": self.suppressed_direction,
                "total": self.total,
                "frames": self.n_frames}

    def _is_direction_allowed(self, dx: float, dy: float, zone_idx: int = 0) -> bool:
        """True if movement (dx, dy) aligns within flow_max_angle_deg of allowed flows."""
        zone = self.zones[zone_idx] if 0 <= zone_idx < len(self.zones) else None
        raw_dirs = getattr(zone, "allowed_directions", None) if zone is not None else None
        if raw_dirs is None and 0 <= zone_idx < len(self._zone_allowed_directions):
            raw_dirs = self._zone_allowed_directions[zone_idx]
        if raw_dirs is None:
            raw_dirs = self.allowed_directions
        target_dirs = _normalize_directions(raw_dirs)
        if not target_dirs:
            return True
        norm = math.hypot(dx, dy)
        if norm <= 1e-9:
            return False
        angle = getattr(zone, "flow_max_angle_deg", None) if zone is not None else None
        if angle is None and 0 <= zone_idx < len(self._zone_flow_max_angle_deg):
            angle = self._zone_flow_max_angle_deg[zone_idx]
        if angle is None:
            angle = self.flow_max_angle_deg
        ux, uy = dx / norm, dy / norm
        cos_thr = math.cos(math.radians(float(angle)))
        return max(ux * vx + uy * vy for vx, vy in target_dirs) >= cos_thr

    def _reattribute(self, zi: int, tid: int, vote: int, conf: float,
                     cx: float, cy: float, now: float) -> Optional[dict]:
        """Shift an already-counted id's class bucket when its stabilized
        class changes while it is still inside the zone (pickup: truck on the
        boundary, car mid-zone). One id = one count: `total` never moves, the
        old bucket decrements (floored at 0), the new one increments, and a
        re-attribute event is returned so hourly/delta reporting stays true.
        A brief non-motor flicker that falls back to the registered class is
        ignored entirely."""
        old_label = self.counted_id_class[zi][tid]
        new_label = self._label(vote)
        if new_label == old_label:
            return None
        # A counted car locked as mobil never becomes a truck/bus: box growth
        # near the bottom or mid-turn is perspective inflation, not a truck.
        if (new_label in self.truck_labels and old_label == self._label(self.CAR_CLS)
                and self._car_locked(tid)):
            return None
        if new_label == self._label(self.CAR_CLS):
            self._car_lock_latched.add(tid)
        self.total_by_class[old_label] = max(0, self.total_by_class[old_label] - 1)
        self.total_by_class[new_label] += 1
        self.counted_id_class[zi][tid] = new_label
        return {"zone": zi, "track_id": tid, "class_id": vote,
                "class_name": new_label, "confidence": conf,
                "cx": cx, "cy": cy, "t": now, "reattributed_from": old_label}

    @staticmethod
    def _inside_truck(car: np.ndarray, trucks: List[np.ndarray],
                      iou_thr: float, area_ratio: float) -> bool:
        """True if a car box is the cabin of a truck.

        Centroid-in-box: only fires when car is also smaller than the truck
        (area_ratio guard). This prevents suppressing a real car that is
        directly behind a bus in dense traffic — same-size vehicles are not
        cabins even if one centroid overlaps the other box.
        IoU path is unchanged: heavy overlap regardless of size = cabin."""
        cx = (car[0] + car[2]) * 0.5
        cy = (car[1] + car[3]) * 0.5
        ca = (car[2] - car[0]) * (car[3] - car[1])
        for t in trucks:
            ta = (t[2] - t[0]) * (t[3] - t[1])
            if (t[0] <= cx <= t[2] and t[1] <= cy <= t[3]
                    and ca < area_ratio * ta):
                return True
            ix1, iy1 = max(car[0], t[0]), max(car[1], t[1])
            ix2, iy2 = min(car[2], t[2]), min(car[3], t[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter / max(ca + ta - inter, 1e-6) >= iou_thr:
                return True
        return False

    def _is_jitter_duplicate(self, zi: int, pt: Tuple[float, float], now: float) -> bool:
        """True if an accepted count < dedup_cooldown ago sits within
        dedup_radius of pt. Expired events pruned in the same pass."""
        events = self._recent[zi]
        while events and now - events[0][0] > self.dedup_cooldown:
            events.popleft()
        return any((ex - pt[0]) ** 2 + (ey - pt[1]) ** 2 <= self.dedup_r2
                   for _, ex, ey in events)

    def _match_ghost(self, zi: int, tid: int, live: set) -> Optional[int]:
        """Nearest unclaimed ghost in zone `zi` that `tid` may inherit.

        Three invariants, all required:
          a) non-coexistence — `tid` was born strictly after the ghost was last
             seen, so the two ids never shared a frame. Two bikes riding
             side by side can therefore never steal each other's latch.
          b) temporal proximity — the birth is within ghost_ttl_frames of the
             ghost's last frame (an occlusion gap, not a new vehicle minutes on).
          c) spatial proximity — the BIRTH centroid is within ghost_radius_px of
             the ghost's last in-zone centroid. Birth, not current position: a
             real second rider is tracked from far away and only arrives at the
             spot later, while an occluded vehicle respawns on top of itself.
        """
        ghosts = self._zone_ghost[zi]
        born = self._born_frame.get(tid)
        birth = self._birth_centroid.get(tid)
        if not ghosts or born is None or birth is None:
            return None
        new_is_motor = (self._resolved_class(tid, -1) == self.MOTOR_CLS)
        motor_label = self._label(self.MOTOR_CLS)
        best, best_d2 = None, self.ghost_r2
        for old, (ox, oy) in ghosts.items():
            if old == tid or old in live:
                continue
            gap = born - self._last_seen_frame.get(old, born)
            if not 1 <= gap <= self.ghost_ttl_frames:   # (a) and (b)
                continue
            old_lbl = self.counted_id_class[zi].get(old)
            old_is_motor = (old_lbl == motor_label if old_lbl is not None
                            else self._resolved_class(old, -1) == self.MOTOR_CLS)
            if new_is_motor != old_is_motor:
                continue
            d2 = (birth[0] - ox) ** 2 + (birth[1] - oy) ** 2
            if d2 <= best_d2:                           # (c), nearest wins
                best, best_d2 = old, d2
        return best

    def _handover(self, zi: int, old: int, new: int) -> None:
        """Move `old`'s counted latch and class state onto `new`, then retire
        the ghost. Same physical vehicle, so `total` and the class buckets do
        not move — only the id that owns the existing count changes."""
        label = self.counted_id_class[zi].pop(old, None)
        self._zone_ghost[zi].pop(old, None)   # consume: never claimed twice
        self._counted[zi].add(new)
        if label is not None:
            self.counted_id_class[zi][new] = label
            if label == self._label(self.CAR_CLS):
                self._car_lock_latched.add(new)
        if old in self._car_lock_latched:
            self._car_lock_latched.add(new)
        if old in self._truck_lock_latched:
            self._truck_lock_latched.add(new)
            self._car_lock_latched.discard(new)
        if old in self._truck_frames:
            self._truck_frames[new] = self._truck_frames.get(new, 0) + self._truck_frames[old]
        if old in self._consec_valid_truck:
            self._consec_valid_truck[new] = max(
                self._consec_valid_truck.get(new, 0), self._consec_valid_truck[old]
            )
        if old in self._peak_truck_area:
            self._peak_truck_area[new] = max(
                self._peak_truck_area.get(new, 0.0), self._peak_truck_area[old]
            )
        if old in self._last_area:
            self._last_area.setdefault(new, self._last_area[old])
        if old in self._last_truck_cls:
            self._last_truck_cls.setdefault(new, self._last_truck_cls[old])
        # Class history carries over (oldest entries drop first) so the vote,
        # motor/car inertia and _reattribute keep working on the new id.
        oh = self.class_hist.get(old)
        if oh:
            own = list(self.class_hist.get(new) or ())
            nh = self.class_hist.setdefault(new, deque(maxlen=self.class_window))
            nh.clear()
            nh.extend(list(oh) + own)
        if old in self._first_cls:      # keeps _car_locked truthful
            self._first_cls[new] = list(self._first_cls[old])
        if old in self._start:          # keeps the displacement anchor
            self._start[new] = self._start[old]
        self._age[new] = self._age.get(new, 0) + self._age.get(old, 0)
        self.ghost_handovers += 1

    def _prune_stale_tracks(self) -> None:
        """Drop expired ghosts every frame and prune stale track state every 100 frames."""
        with self._zone_lock:
            ghost_cutoff = self._frame_idx - self.ghost_ttl_frames
            for ghosts in self._zone_ghost:
                for old in [o for o in ghosts if self._last_seen_frame.get(o, 0) < ghost_cutoff]:
                    ghosts.pop(old, None)
            if self._frame_idx % 100 != 0:
                return
            stale_cutoff = self._frame_idx - max(self.max_stale_frames, self.ghost_ttl_frames * 3)
            stale_tids = [tid for tid, last_seen in self._last_seen_frame.items()
                          if last_seen < stale_cutoff]
            for tid in stale_tids:
                self.class_hist.pop(tid, None)
                self.trails.pop(tid, None)
                self._prev.pop(tid, None)
                self._start.pop(tid, None)
                self._first_cls.pop(tid, None)
                self._car_lock_latched.discard(tid)
                self._truck_lock_latched.discard(tid)
                self._truck_frames.pop(tid, None)
                self._consec_valid_truck.pop(tid, None)
                self._peak_truck_area.pop(tid, None)
                self._last_area.pop(tid, None)
                self._last_truck_cls.pop(tid, None)
                self._zone_still.pop(tid, None)
                self._age.pop(tid, None)
                self._missing.pop(tid, None)
                self._born_frame.pop(tid, None)
                self._birth_centroid.pop(tid, None)
                self._last_seen_frame.pop(tid, None)
                self._last_centroid.pop(tid, None)
                for zi in range(len(self.zones)):
                    self._counted[zi].discard(tid)
                    self.counted_id_class[zi].pop(tid, None)
                    self._zone_ghost[zi].pop(tid, None)

    def update(self, dets: sv.Detections, now: Optional[float] = None,
               freeze: bool = False) -> List[dict]:
        """Return count events registered this frame:
        {zone, track_id, class_id, class_name, confidence, cx, cy, t}.
        freeze=True (stream jitter grace): keep debug state current but do
        not register counts while the tracker re-stabilizes its ids."""
        with self._zone_lock:
            return self._update_locked(dets, now=now, freeze=freeze)

    def _update_locked(self, dets: sv.Detections, now: Optional[float] = None,
                       freeze: bool = False) -> List[dict]:
        if now is None:
            now = time.monotonic()
        self.n_frames += 1
        self._frame_idx += 1   # counts empty frames too: occlusion gaps are frames
        events = []
        if len(dets) == 0 or dets.tracker_id is None:
            self._prune_stale_tracks()
            return events
        cents = centroids(dets.xyxy)
        tids = dets.tracker_id.astype(int)
        live = set(int(t) for t in tids)
        # contours rebuilt per frame: ZoneEditor drags polygon points live
        polys = [np.asarray(z.polygon, dtype=np.float64) for z in self.zones]
        contours = [p.reshape(-1, 1, 2).astype(np.int32) for p in polys]
        # truck/bus boxes this frame, for cabin suppression (raw class: a truck
        # flickering to car for one frame is rare and cheap to miss)
        truck_boxes = ([dets.xyxy[j] for j in range(len(dets))
                        if int(dets.class_id[j]) in self.TRUCK_CLS]
                       if dets.class_id is not None else [])
        # Rows the detector demoted truck->car by size (detector.py) carry a
        # flag in Detections.data. A demoted pickup shown as class 2 would
        # otherwise be swallowed by the cabin filter if another truck box
        # overlaps it — but it is a whole vehicle, never a truck cabin.
        flags = dets.data.get("reclassified") if dets.data else None
        if flags is None or len(flags) != len(dets):
            flags = None
        # Rows the detector demoted car->motor by size (detector.py) keep
        # their cabin candidacy: a small car box inside a truck box is the
        # truck cabin even after the sweep relabels it a motorcycle.
        car_demoted = dets.data.get("demoted_car") if dets.data else None
        if car_demoted is None or len(car_demoted) != len(dets):
            car_demoted = None
        persp_trucks = dets.data.get("perspective_truck") if dets.data else None
        if persp_trucks is None or len(persp_trucks) != len(dets):
            persp_trucks = None

        for i in range(len(dets)):
            tid = int(tids[i])
            cls = int(dets.class_id[i]) if dets.class_id is not None else -1
            w_box = max(0.0, float(dets.xyxy[i][2] - dets.xyxy[i][0]))
            h_box = max(0.0, float(dets.xyxy[i][3] - dets.xyxy[i][1]))
            area = w_box * h_box
            aspect = w_box / max(h_box, 1e-6)
            pt = (float(cents[i][0]), float(cents[i][1]))
            bottom_pt = (float(cents[i][0]), float(dets.xyxy[i][3]))
            conf = float(dets.confidence[i]) if dets.confidence is not None else -1.0
            prev = self._prev.get(tid)
            self._prev[tid] = pt
            self._last_area[tid] = area
            # Displacement anchor on the FIRST frame the id appears anywhere in
            # the view, not on zone entry: a fast bike tracked for a frame or
            # two outside the polygon still clears min_displacement_px on the
            # single frame it lands inside, while a roadside pot jittering on
            # the boundary never drifts from its birth centroid.
            self._start.setdefault(tid, pt)
            if tid not in self._born_frame:
                self._born_frame[tid] = self._frame_idx
                self._birth_centroid[tid] = pt
            self._last_seen_frame[tid] = self._frame_idx
            self._last_centroid[tid] = pt
            self._age[tid] = self._age.get(tid, 0) + 1
            fc = self._first_cls.setdefault(tid, [])
            if len(fc) < 2:
                fc.append(cls)
            self._missing.pop(tid, None)
            h = self.class_hist.setdefault(tid, deque(maxlen=self.class_window))
            h.append((cls, conf))
            if cls in self.TRUCK_CLS:
                self._last_truck_cls[tid] = cls
                self._truck_frames[tid] = self._truck_frames.get(tid, 0) + 1
                self._peak_truck_area[tid] = max(self._peak_truck_area.get(tid, 0.0), area)
                aspect_ok = aspect <= self.truck_promote_max_aspect
                is_persp = persp_trucks is not None and bool(persp_trucks[i])
                is_large = area >= self.truck_promote_area
                is_valid_truck = aspect_ok and (is_persp or is_large)
                self._consec_valid_truck[tid] = (
                    self._consec_valid_truck.get(tid, 0) + 1 if is_valid_truck else 0
                )
                if tid in self._car_lock_latched or (fc and fc[0] == self.CAR_CLS):
                    # Distant approaching medium truck: >=2 consecutive
                    # perspective-validated TRUCK_CLS frames promote car -> truck.
                    if self._consec_valid_truck[tid] >= 2:
                        self._car_lock_latched.discard(tid)
                        self._truck_lock_latched.add(tid)
                elif self._truck_frames[tid] >= 2 or (
                    self._zone_y_mid is not None
                    and bottom_pt[1] >= self._zone_y_mid
                    and is_large
                    and aspect_ok
                ):
                    self._truck_lock_latched.add(tid)
            else:
                self._consec_valid_truck[tid] = 0
            if tid not in self._truck_lock_latched and (
                (fc and fc[0] == self.CAR_CLS and self._age[tid] >= 2)
                or (
                    not (fc and fc[0] in self.TRUCK_CLS)
                    and sum(1 for c, _ in h if c == self.CAR_CLS) >= self.car_lock_frames
                )
            ):
                self._car_lock_latched.add(tid)
            if self.debug:
                tr = self.trails.setdefault(tid, deque(maxlen=self.TRAIL_LEN))
                tr.append(pt)
            if freeze or tid < 0 or self._age[tid] < self.min_track_frames:
                continue
            # A stable, confident id outranks spatial dedup: swarm riders
            # passing the same spot are real, not re-ID ghosts.
            trusted = (self._age[tid] >= self.STABLE_FRAMES
                       and conf >= self.STABLE_CONF)
            for zi, contour in enumerate(contours):
                in_zone = (cv2.pointPolygonTest(contour, pt, False) >= 0
                           or cv2.pointPolygonTest(contour, bottom_pt, False) >= 0)
                if tid in self._counted[zi]:
                    # Already counted here: re-vote the class while the id is
                    # still active inside the zone. Pickup mid-zone flips
                    # (truck -> car) re-attribute the count instead of being
                    # lost; a one-frame flicker that votes back to the
                    # registered class is a no-op.
                    if tid in self.counted_id_class[zi] and in_zone:
                        # Refresh this id's ghost every frame it is counted and
                        # inside: if it vanishes next frame (occlusion, RTSP
                        # micro-stutter) its respawned id inherits the latch.
                        self._zone_ghost[zi][tid] = pt
                        re = self._reattribute(zi, tid,
                                               self._resolved_class(tid, cls),
                                               conf, pt[0], pt[1], now)
                        if re is not None:
                            events.append(re)
                    continue
                inside = in_zone
                if not inside and not (prev is not None
                                       and _crosses_polygon(polys[zi], prev, pt)):
                    continue
                # Inherit pre-occlusion displacement/flow anchor if a ghost matches.
                ghost = self._match_ghost(zi, tid, live)
                if ghost is not None and ghost in self._start:
                    self._start[tid] = self._start[ghost]
                # Filter 1: displacement gate — static FP (bush) never counts.
                # Escape hatch: a vehicle stopped in a jam is real; after
                # still_count_frames inside the zone with conf >= still_min_conf
                # it counts regardless (city rush-hour / intersection queue case).
                sx, sy = self._start[tid]
                if (pt[0] - sx) ** 2 + (pt[1] - sy) ** 2 < self.min_disp2:
                    still = self._zone_still.setdefault(tid, {})
                    still[zi] = still.get(zi, 0) + 1
                    if (still[zi] >= self.still_count_frames
                            and conf >= self.still_min_conf):
                        still.pop(zi)  # reset so re-entry after exit doesn't skip
                    else:
                        continue
                if not self._is_direction_allowed(pt[0] - sx, pt[1] - sy, zi):
                    self.suppressed_direction += 1
                    continue
                vote = self._resolved_class(tid, cls)
                # Filter 3: a car inside a truck box is the truck cabin. A
                # size-demoted truck (flags[i]) is exempt — it only reads as a
                # car, so it must never be discarded as another truck's cabin.
                # A car demoted to motor by the car_min_area sweep is still a
                # cabin candidate when its box sits inside a truck box.
                cabin_box = (vote == self.CAR_CLS
                             or (car_demoted is not None
                                 and bool(car_demoted[i])))
                other_trucks = ([dets.xyxy[j] for j in range(len(dets))
                                 if j != i and int(dets.class_id[j]) in self.TRUCK_CLS]
                                if dets.class_id is not None else [])
                if (cabin_box and other_trucks
                        and not (flags is not None and bool(flags[i]))
                        and self._inside_truck(dets.xyxy[i], other_trucks,
                                               self.cabin_iou,
                                               self.cabin_area_ratio)):
                    self._counted[zi].add(tid)   # latch: never re-trigger
                    self.suppressed_cabin += 1
                    continue
                if not trusted and self._is_jitter_duplicate(zi, pt, now):
                    self._counted[zi].add(tid)
                    self.suppressed += 1
                    # The jitter filter just aliased this id to the counted
                    # vehicle at that spot, so retire that vehicle's ghost: the
                    # handover is spent. A later id near the same spot is then
                    # a genuinely different vehicle and counts normally.
                    for old in [o for o, (ox, oy) in self._zone_ghost[zi].items()
                                if (ox - pt[0]) ** 2 + (oy - pt[1]) ** 2 <= self.dedup_r2]:
                        self._zone_ghost[zi].pop(old, None)
                    continue
                # Ghost handover: a vehicle already counted in this zone that
                # vanished (occlusion / RTSP micro-stutter) and respawned under
                # a new tracker_id is the same vehicle — inherit its latch and
                # class state instead of counting it a second time. Checked
                # after the cabin and jitter filters so their counters, and the
                # one-frame re-ID suppression they own, keep their meaning.
                ghost = self._match_ghost(zi, tid, live)
                if ghost is not None:
                    self._handover(zi, ghost, tid)
                    continue
                self._counted[zi].add(tid)
                label = self._label(vote)
                self.counted_id_class[zi][tid] = label
                self._recent[zi].append((now, pt[0], pt[1]))
                self.total += 1
                self.total_by_class[label] += 1
                events.append({"zone": zi, "track_id": tid, "class_id": vote,
                               "class_name": label, "confidence": conf,
                               "cx": pt[0], "cy": pt[1], "t": now})
        for tid in [t for t in self.class_hist if t not in live]:
            self._missing[tid] = self._missing.get(tid, 0) + 1
            if self._missing[tid] >= self.class_window:
                self.class_hist.pop(tid, None)
                self._missing.pop(tid, None)
        self._prune_stale_tracks()
        return events

    def reset(self) -> None:
        """Zero counts, latches, and the jitter log."""
        with self._zone_lock:
            self.total = 0
            self.suppressed = 0
            self.suppressed_cabin = 0
            self.suppressed_direction = 0
            self.total_by_class.clear()
            self.class_hist.clear()
            self.trails.clear()
            self._prev.clear()
            self._start.clear()
            self._first_cls.clear()
            self._car_lock_latched.clear()
            self._truck_lock_latched.clear()
            self._truck_frames.clear()
            self._consec_valid_truck.clear()
            self._peak_truck_area.clear()
            self._last_area.clear()
            self._last_truck_cls.clear()
            self._zone_still.clear()
            self._age.clear()
            self._missing.clear()
            self._born_frame.clear()
            self._birth_centroid.clear()
            self._last_seen_frame.clear()
            self._last_centroid.clear()
            self._zone_ghost = [dict() for _ in self.zones]
            self._frame_idx = 0
            self.ghost_handovers = 0
            self._counted = [set() for _ in self.zones]
            self.counted_id_class = [dict() for _ in self.zones]
            self._recent = [deque() for _ in self.zones]

    def update_polygon(self, normalized_points, frame_width: int,
                       frame_height: int, zone_index: int = 0) -> None:
        """Swap zone `zone_index` for a polygon drawn on the web dashboard.

        `normalized_points` are 0.0-1.0 ratios of the raw frame (the frontend
        letterbox-maps clicks to the video box first), so they scale to any
        stream resolution. Replaces the sv.PolygonZone in place; the latches
        reset so vehicles inside the new polygon re-qualify."""
        pts = np.asarray(normalized_points, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 3:
            raise ValueError("polygon needs >= 3 points")
        pixel_pts = pts * [float(frame_width), float(frame_height)]
        with self._zone_lock:
            old_zone = self.zones[zone_index]
            new_zone = sv.PolygonZone(polygon=pixel_pts.astype(np.float32))
            for attr in ("allowed_directions", "flow_max_angle_deg"):
                if hasattr(old_zone, attr):
                    setattr(new_zone, attr, getattr(old_zone, attr))
            self.zones[zone_index] = new_zone
            self._zone_y_mid = self._compute_zone_y_mid()
            self._counted = [set() for _ in self.zones]
            self.counted_id_class = [dict() for _ in self.zones]
            self._recent = [deque() for _ in self.zones]
            self._zone_ghost = [dict() for _ in self.zones]   # latches are gone

    def set_zones(self, zones: List[Union[sv.PolygonZone, dict]]) -> None:
        """Replace all zone polygons in place while preserving counts and active tracks."""
        new_zones: List[sv.PolygonZone] = []
        new_ads = []
        new_angs = []
        for idx, z in enumerate(zones):
            old = self.zones[idx] if idx < len(self.zones) else None
            if isinstance(z, dict):
                pts = np.asarray(z["polygon"], dtype=np.float32).reshape(-1, 2)
                if pts.shape[0] < 3:
                    raise ValueError("polygon needs >= 3 points")
                pz = sv.PolygonZone(polygon=pts)
                ad = z.get("allowed_directions")
                if ad is None and old is not None:
                    ad = getattr(old, "allowed_directions", None)
                ang = z.get("flow_max_angle_deg")
                if ang is None and old is not None:
                    ang = getattr(old, "flow_max_angle_deg", None)
                if ad is not None:
                    _normalize_directions(ad)
                    pz.allowed_directions = ad
                if ang is not None:
                    pz.flow_max_angle_deg = ang
                new_zones.append(pz)
                new_ads.append(ad)
                new_angs.append(ang)
            else:
                pts = np.asarray(z.polygon, dtype=np.float32).reshape(-1, 2)
                if pts.shape[0] < 3:
                    raise ValueError("polygon needs >= 3 points")
                ad = getattr(z, "allowed_directions", None)
                if ad is None and old is not None:
                    ad = getattr(old, "allowed_directions", None)
                    if ad is not None:
                        z.allowed_directions = ad
                ang = getattr(z, "flow_max_angle_deg", None)
                if ang is None and old is not None:
                    ang = getattr(old, "flow_max_angle_deg", None)
                    if ang is not None:
                        z.flow_max_angle_deg = ang
                if ad is not None:
                    _normalize_directions(ad)
                new_zones.append(z)
                new_ads.append(ad)
                new_angs.append(ang)
        with self._zone_lock:
            self.zones = new_zones
            self._zone_allowed_directions = new_ads
            self._zone_flow_max_angle_deg = new_angs
            self._zone_y_mid = self._compute_zone_y_mid()
            self._counted = [set() for _ in self.zones]
            self.counted_id_class = [dict() for _ in self.zones]
            self._recent = [deque() for _ in self.zones]
            self._zone_ghost = [dict() for _ in self.zones]
