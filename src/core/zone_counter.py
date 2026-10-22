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
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from .detector import centroids


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

    def __init__(self, zones: List[sv.PolygonZone],
                 class_labels: Optional[Dict[int, str]] = None,
                 tracker_config: Optional[dict] = None,
                 dedup_cooldown_sec: float = 4.0, dedup_radius_px: float = 10.0,
                 min_displacement_px: float = 30.0, class_window: int = CLASS_WINDOW,
                 motor_lock_frames: int = MOTOR_LOCK_FRAMES,
                 motor_flip_frames: int = MOTOR_FLIP_FRAMES,
                 car_lock_frames: int = CAR_LOCK_FRAMES,
                 car_flip_frames: int = CAR_FLIP_FRAMES,
                 cabin_iou: float = 0.6, cabin_area_ratio: float = 0.55,
                 still_count_frames: int = 30, still_min_conf: float = 0.35,
                 debug: bool = False):
        from trackers import CBIoUTracker

        self.zones = zones
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
        self.debug = bool(debug)
        self.tracker = CBIoUTracker(**(tracker_config or {
            "lost_track_buffer": 90,
            "track_activation_threshold": 0.20,
            "minimum_consecutive_frames": 1,
        }))
        self.total = 0
        self.suppressed = 0
        self.suppressed_cabin = 0
        self.total_by_class = defaultdict(int)
        self.n_frames = 0
        self.class_hist = {}   # tid -> deque of (class_id, confidence)
        # counted_id_class[zi][tid] = label registered when this id first
        # counted in zone zi. Drives atomic re-attribution: a counted id whose
        # stabilized class drifts while still active in the zone moves its
        # count from the old class bucket to the new one (total unchanged).
        self.counted_id_class = [dict() for _ in zones]
        self.trails = {}       # tid -> deque of centroid pts (debug render)
        self._first_cls = {}   # tid -> first 2 class ids, for the car lock
        self._prev = {}        # tid -> last centroid, for jump-over-edge sweep
        self._start = {}       # tid -> first centroid, for the displacement gate
        self._age = {}         # tid -> frames seen, for the stable-id trust rule
        self._missing = {}     # tid -> frames since last seen (hist eviction)
        self._zone_still = {}  # tid -> {zi: frames blocked by displacement gate}
        self._counted = [set() for _ in zones]   # permanent per-zone id latch
        self._recent = [deque() for _ in zones]  # (t, x, y) accepted count events

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
        if (vote in self.TRUCK_CLS and self._car_locked(tid)
                and self._longest_non_run(h, self.CAR_CLS) <= self.car_flip_frames):
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
        """True once a track opened as car (not truck/bus in its first 2
        frames) and accumulated car_lock_frames car sightings."""
        if any(c in self.TRUCK_CLS for c in self._first_cls.get(tid, ())):
            return False
        h = self.class_hist.get(tid)
        return bool(h) and sum(1 for c, _ in h if c == self.CAR_CLS) >= self.car_lock_frames

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
        # A counted car that accumulated car_lock_frames car sightings never
        # becomes a truck: the flip is the box inflating near the camera or
        # mid-turn, and moving the count would invent a truck that never came.
        if (new_label in self.truck_labels and old_label == self._label(self.CAR_CLS)
                and self._car_locked(tid)):
            return None
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

    def update(self, dets: sv.Detections, now: Optional[float] = None,
               freeze: bool = False) -> List[dict]:
        """Return count events registered this frame:
        {zone, track_id, class_id, class_name, confidence, cx, cy, t}.
        freeze=True (stream jitter grace): keep debug state current but do
        not register counts while the tracker re-stabilizes its ids."""
        if now is None:
            now = time.monotonic()
        self.n_frames += 1
        events = []
        if len(dets) == 0 or dets.tracker_id is None:
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

        for i in range(len(dets)):
            tid = int(tids[i])
            cls = int(dets.class_id[i]) if dets.class_id is not None else -1
            pt = (float(cents[i][0]), float(cents[i][1]))
            conf = float(dets.confidence[i]) if dets.confidence is not None else -1.0
            prev = self._prev.get(tid)
            self._prev[tid] = pt
            self._start.setdefault(tid, pt)
            self._age[tid] = self._age.get(tid, 0) + 1
            fc = self._first_cls.setdefault(tid, [])
            if len(fc) < 2:
                fc.append(cls)
            self._missing.pop(tid, None)
            h = self.class_hist.setdefault(tid, deque(maxlen=self.class_window))
            h.append((cls, conf))
            if self.debug:
                tr = self.trails.setdefault(tid, deque(maxlen=self.TRAIL_LEN))
                tr.append(pt)
            if freeze or tid < 0:
                continue
            # A stable, confident id outranks spatial dedup: swarm riders
            # passing the same spot are real, not re-ID ghosts.
            trusted = (self._age[tid] >= self.STABLE_FRAMES
                       and conf >= self.STABLE_CONF)
            for zi, contour in enumerate(contours):
                if tid in self._counted[zi]:
                    # Already counted here: re-vote the class while the id is
                    # still active inside the zone. Pickup mid-zone flips
                    # (truck -> car) re-attribute the count instead of being
                    # lost; a one-frame flicker that votes back to the
                    # registered class is a no-op.
                    if (tid in self.counted_id_class[zi]
                            and cv2.pointPolygonTest(contour, pt, False) >= 0):
                        re = self._reattribute(zi, tid,
                                               self._resolved_class(tid, cls),
                                               conf, pt[0], pt[1], now)
                        if re is not None:
                            events.append(re)
                    continue
                inside = cv2.pointPolygonTest(contour, pt, False) >= 0
                if not inside and not (prev is not None
                                       and _crosses_polygon(polys[zi], prev, pt)):
                    continue
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
                vote = self._resolved_class(tid, cls)
                # Filter 3: a car inside a truck box is the truck cabin. A
                # size-demoted truck (flags[i]) is exempt — it only reads as a
                # car, so it must never be discarded as another truck's cabin.
                # A car demoted to motor by the car_min_area sweep is still a
                # cabin candidate when its box sits inside a truck box.
                cabin_box = (vote == self.CAR_CLS
                             or (car_demoted is not None
                                 and bool(car_demoted[i])))
                if (cabin_box and truck_boxes
                        and not (flags is not None and bool(flags[i]))
                        and self._inside_truck(dets.xyxy[i], truck_boxes,
                                               self.cabin_iou,
                                               self.cabin_area_ratio)):
                    self._counted[zi].add(tid)   # latch: never re-trigger
                    self.suppressed_cabin += 1
                    continue
                if not trusted and self._is_jitter_duplicate(zi, pt, now):
                    self._counted[zi].add(tid)
                    self.suppressed += 1
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
                self.trails.pop(tid, None)
                self._prev.pop(tid, None)
                self._start.pop(tid, None)
                self._first_cls.pop(tid, None)
                self._zone_still.pop(tid, None)
                self._age.pop(tid, None)
                self._missing.pop(tid, None)
        return events

    def reset(self) -> None:
        """Zero counts, latches, and the jitter log."""
        self.total = 0
        self.suppressed = 0
        self.suppressed_cabin = 0
        self.total_by_class.clear()
        self._prev.clear()
        self._start.clear()
        self._first_cls.clear()
        self._zone_still.clear()
        self._age.clear()
        self._missing.clear()
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
        self.zones[zone_index] = sv.PolygonZone(
            polygon=pixel_pts.astype(np.float32))
        self._counted = [set() for _ in self.zones]
        self.counted_id_class = [dict() for _ in self.zones]
        self._recent = [deque() for _ in self.zones]
