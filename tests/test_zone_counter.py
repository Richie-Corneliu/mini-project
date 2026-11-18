"""Core-logic checks for cross_class_nms + ZoneCounter (no GPU needed).
Run:  python -m tests.test_zone_counter  (repo root on PYTHONPATH)"""
import csv
import os
import tempfile
import time

import numpy as np
import supervision as sv

from src.core.detector import (apply_size_heuristics, cross_class_nms,
                               reclassify_small_cars, reclassify_small_trucks)
from src.core.zone_counter import ZoneCounter
from src.utils.logger import AuditLogger


def _dets(xyxy, cls, conf=None, tid=None, reclassified=None, demoted_car=None):
    n = len(xyxy)
    d = sv.Detections(
        xyxy=np.asarray(xyxy, dtype=np.float32),
        confidence=np.asarray(conf if conf is not None else [0.9] * n, dtype=np.float32),
        class_id=np.asarray(cls, dtype=np.int32),
        tracker_id=np.asarray(tid, dtype=np.int64) if tid is not None else None,
    )
    if reclassified is not None:
        d.data["reclassified"] = np.asarray(reclassified, dtype=bool)
    if demoted_car is not None:
        d.data["demoted_car"] = np.asarray(demoted_car, dtype=bool)
    return d


def test_nms_car_beats_motorcycle():
    d = _dets([[0, 0, 100, 100], [10, 10, 110, 110]], [3, 2])  # motor first, car bigger-priority
    kept = cross_class_nms(d, duplicate_iou=0.5)
    assert list(kept.class_id) == [2], kept.class_id


def test_nms_tie_keeps_bigger():
    d = _dets([[0, 0, 100, 100], [5, 5, 60, 60]], [2, 2])
    kept = cross_class_nms(d, duplicate_iou=0.3)
    assert len(kept) == 1
    assert kept.xyxy[0][2] == 100


def test_small_truck_demoted_to_car():
    # 60x60=3600 < 8000 -> car; 200x200=40000 stays truck; car/motor untouched
    d = _dets([[0, 0, 60, 60], [0, 100, 200, 300],
               [0, 400, 40, 440], [0, 500, 50, 550]],
              [7, 7, 2, 3])
    out = reclassify_small_trucks(d, truck_min_area=8000)
    assert list(out.class_id) == [2, 7, 2, 3], out.class_id


def test_reclassify_disabled_and_boundary():
    d = _dets([[0, 0, 10, 10]], [7])
    assert list(reclassify_small_trucks(d, truck_min_area=0).class_id) == [7]
    # exact threshold is NOT below -> stays truck
    d2 = _dets([[0, 0, 100, 100]], [7])   # area == 10000
    assert list(reclassify_small_trucks(d2, truck_min_area=10000).class_id) == [7]


def test_reclassify_empty_dets():
    d = sv.Detections.empty()
    assert reclassify_small_trucks(d, truck_min_area=8000) is d


def test_zone_latch_and_dedup():
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [200, 0], [200, 200], [0, 200]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil"}, dedup_radius_px=40,
                     dedup_cooldown_sec=4.0, min_displacement_px=0, min_track_frames=1)
    t = time.monotonic()
    d = _dets([[90, 90, 110, 110]], [2], tid=[1])
    ev = zc.update(d, now=t)
    assert len(ev) == 1 and zc.total == 1
    ev = zc.update(d, now=t + 1)  # same id still inside -> latch
    assert not ev and zc.total == 1
    # re-ID 1px away = jitter duplicate -> suppressed, aliased into latch
    d2 = _dets([[91, 91, 111, 111]], [2], tid=[2])
    ev = zc.update(d2, now=t + 2)
    assert not ev and zc.total == 1 and zc.suppressed == 1
    # fresh id far away or after cooldown -> counts
    d3 = _dets([[91, 91, 111, 111]], [2], tid=[3])
    ev = zc.update(d3, now=t + 10)
    assert len(ev) == 1 and zc.total == 2


def test_class_voting():
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [200, 0], [200, 200], [0, 200]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 3: "motor"})
    t = time.monotonic()
    outside = [[300, 300, 340, 340]]
    inside = [[95, 95, 105, 105]]
    for i in range(4):  # stable track, history flickers motor x4... outside zone
        zc.update(_dets(outside, [3], tid=[1]), now=t + i)
    for i in range(8):  # ...then mobil x8, still outside (window holds 15)
        zc.update(_dets(outside, [2], tid=[1]), now=t + 4 + i)
    # crossing frame flickers back to motor; window majority says mobil
    ev = zc.update(_dets(inside, [3], tid=[1]), now=t + 20)
    assert len(ev) == 1 and ev[0]["class_name"] == "mobil", ev


def test_class_voting_tie_prefers_conf():
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [200, 0], [200, 200], [0, 200]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 3: "motor"})
    t = time.monotonic()
    outside = [[300, 300, 340, 340]]
    inside = [[95, 95, 105, 105]]
    for i in range(4):
        zc.update(_dets(outside, [3], conf=[0.3], tid=[1]), now=t + i)
    for i in range(5):
        zc.update(_dets(outside, [2], conf=[0.9], tid=[1]), now=t + 4 + i)
    # 5-5 tie in the window; mobil has higher summed confidence
    ev = zc.update(_dets(inside, [3], conf=[0.3], tid=[1]), now=t + 20)
    assert len(ev) == 1 and ev[0]["class_name"] == "mobil", ev

def test_hist_evicted_after_gap():
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [200, 0], [200, 200], [0, 200]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 3: "motor"})
    t = time.monotonic()
    zc.update(_dets([[300, 300, 340, 340]], [3], tid=[1]), now=t)
    other = _dets([[50, 50, 60, 60]], [2], tid=[9])
    for i in range(ZoneCounter.CLASS_WINDOW):
        zc.update(other, now=t + 1 + i)
    assert 1 not in zc.class_hist

def test_jump_over_polygon_counts():
    zone = sv.PolygonZone(polygon=np.array([[100, 200], [600, 200], [600, 500], [100, 500]]))
    zc = ZoneCounter([zone], class_labels={3: "motor"})
    t = time.monotonic()
    zc.update(_dets([[330, 130, 370, 170]], [3], tid=[1]), now=t)      # above
    ev = zc.update(_dets([[330, 530, 370, 570]], [3], tid=[1]), now=t + 1)  # below
    assert len(ev) == 1 and zc.total == 1, ev


def test_parallel_outside_no_count():
    zone = sv.PolygonZone(polygon=np.array([[100, 200], [600, 200], [600, 500], [100, 500]]))
    zc = ZoneCounter([zone], class_labels={3: "motor"})
    t = time.monotonic()
    for k in range(4):
        assert not zc.update(_dets([[50, 130 + k * 10, 90, 170 + k * 10]], [3], tid=[1]),
                             now=t + k)
    assert zc.total == 0


def test_stable_id_beats_spatial_dedup():
    zone = sv.PolygonZone(polygon=np.array([[100, 200], [600, 200], [600, 500], [100, 500]]))
    zc = ZoneCounter([zone], class_labels={3: "motor"}, dedup_radius_px=80,
                     dedup_cooldown_sec=4.0)
    t = time.monotonic()
    for k in range(ZoneCounter.STABLE_FRAMES):  # rider waits far from the zone
        zc.update(_dets([[340, 100, 360, 120]], [3], conf=[0.8], tid=[1]), now=t + k)
    zc.update(_dets([[340, 250, 360, 270]], [3], tid=[1]), now=t + 6)  # counts
    assert zc.total == 1
    for k in range(ZoneCounter.STABLE_FRAMES):
        zc.update(_dets([[540, 100, 560, 120]], [3], conf=[0.8], tid=[2]),
                  now=t + 7 + k)
    # second rider crosses the same spot: within radius, but long-lived -> counts
    ev = zc.update(_dets([[340, 252, 360, 272]], [3], conf=[0.8], tid=[2]), now=t + 13)
    assert zc.total == 2 and len(ev) == 1, f"total={zc.total} sup={zc.suppressed}"


def test_ghost_id_still_suppressed():
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [200, 0], [200, 200], [0, 200]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil"}, dedup_radius_px=10,
                     dedup_cooldown_sec=4.0, min_displacement_px=0, min_track_frames=1)
    t = time.monotonic()
    assert zc.update(_dets([[90, 90, 110, 110]], [2], tid=[1]), now=t)
    assert not zc.update(_dets([[91, 91, 111, 111]], [2], tid=[2]), now=t + 0.2)
    assert zc.total == 1 and zc.suppressed == 1


def test_freeze_blocks_counts():
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [200, 0], [200, 200], [0, 200]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil"})
    d = _dets([[90, 90, 110, 110]], [2], tid=[7])
    assert not zc.update(d, now=time.monotonic(), freeze=True) and zc.total == 0


def test_static_bush_not_counted():
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [200, 0], [200, 200], [0, 200]]))
    zc = ZoneCounter([zone], class_labels={3: "motor"})  # default gate = 30px
    t = time.monotonic()
    for i in range(10):
        assert not zc.update(_dets([[90, 90, 110, 110]], [3], tid=[1]), now=t + i)
    assert zc.total == 0


def test_static_planter_not_counted_as_car():
    """Large roadside planter jittering < 5px as a car in the zone: the
    displacement gate blocks it, even with conf and dwell time."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [200, 0], [200, 200], [0, 200]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil"})
    t = time.monotonic()
    jitter = [(0, 0), (2, 1), (1, 3), (3, 2), (0, 4)]
    for i in range(10):
        dx, dy = jitter[i % len(jitter)]
        box = [[90 + dx, 90 + dy, 130 + dx, 130 + dy]]
        assert not zc.update(_dets(box, [2], conf=[0.9], tid=[1]), now=t + i)
    assert zc.total == 0, zc.total
    assert zc.total_by_class.get("mobil", 0) == 0, dict(zc.total_by_class)


def test_moving_object_counted():
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [400, 0], [400, 400], [0, 400]]))
    zc = ZoneCounter([zone], class_labels={3: "motor"})  # default gate = 30px
    t = time.monotonic()
    zc.update(_dets([[60, 60, 80, 80]], [3], tid=[1]), now=t)               # start (70,70)
    ev = zc.update(_dets([[140, 140, 160, 160]], [3], tid=[1]), now=t + 1)  # moved ~113px
    assert len(ev) == 1 and zc.total == 1, ev


def test_cabin_inside_truck_suppressed():
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [400, 0], [400, 400], [0, 400]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 7: "truk/bus"})
    t = time.monotonic()
    # frame 1: car cabin + whole truck together; car starts the displacement clock
    zc.update(_dets([[80, 80, 120, 120], [70, 70, 210, 210]], [2, 7], tid=[1, 2]), now=t)
    # frame 2: both move; car centroid (140,140) now sits inside the truck box
    ev = zc.update(_dets([[120, 120, 160, 160], [110, 110, 250, 250]], [2, 7],
                         tid=[1, 2]), now=t + 1)
    assert zc.suppressed_cabin == 1, zc.suppressed_cabin
    assert not any(e["track_id"] == 1 for e in ev), ev   # cabin discarded
    assert any(e["track_id"] == 2 for e in ev), ev        # truck still counts


def test_still_vehicle_counts_after_timeout():
    """Stopped vehicle in intersection queue counts after still_count_frames."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [400, 0], [400, 400], [0, 400]]))
    zc = ZoneCounter([zone], class_labels={3: "motor"},
                     min_displacement_px=50, still_count_frames=5, still_min_conf=0.35,
                     min_track_frames=1)
    t = time.monotonic()
    box = [[90, 90, 110, 110]]  # never moves, displacement = 0
    for i in range(4):
        assert not zc.update(_dets(box, [3], tid=[1]), now=t + i), f"frame {i} should not count"
    ev = zc.update(_dets(box, [3], tid=[1]), now=t + 5)
    assert len(ev) == 1 and zc.total == 1, ev


def test_still_low_conf_never_counts():
    """Low-confidence static detection (bush) never fires the still-vehicle escape hatch."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [400, 0], [400, 400], [0, 400]]))
    zc = ZoneCounter([zone], class_labels={3: "motor"},
                     min_displacement_px=50, still_count_frames=5, still_min_conf=0.35,
                     min_track_frames=1)
    t = time.monotonic()
    for i in range(30):
        zc.update(_dets([[90, 90, 110, 110]], [3], conf=[0.1], tid=[1]), now=t + i)
    assert zc.total == 0


def test_same_size_car_not_suppressed_as_cabin():
    """A car the same size as the truck it's beside is not suppressed as a cabin."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [600, 0], [600, 600], [0, 600]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 7: "truk/bus"})
    t = time.monotonic()
    # frame 1: car left-of-truck, start centroid (275,100)
    zc.update(_dets([[150, 0, 400, 200], [0, 0, 400, 200]], [2, 7], tid=[1, 2]), now=t)
    # frame 2: car moves right (disp=100px), centroid (375,100) enters truck x-range.
    # car_area=50000, truck_area=80000, ratio=0.625 > 0.55 → centroid-in-box NOT cabin.
    # IoU=0.30 < 0.6 → IoU path also clear. Car must count.
    ev = zc.update(_dets([[250, 0, 500, 200], [0, 0, 400, 200]], [2, 7],
                         tid=[1, 2]), now=t + 1)
    assert any(e["track_id"] == 1 for e in ev), f"same-size car wrongly suppressed: {ev}"
    assert zc.suppressed_cabin == 0, zc.suppressed_cabin


def _cabin_frame(flagged):
    """Small car-class box whose centroid lands inside a big truck box (area
    ratio 1600/19600 < 0.55, IoU > 0.6) — the exact cabin geometry. Both boxes
    move 50px so the displacement gate passes. `flagged` adds the reclassified
    marker to the car row."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [400, 0], [400, 400], [0, 400]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 7: "truk/bus"},
                     min_displacement_px=50)
    t = time.monotonic()
    mark = [flagged, False] if flagged is not None else None
    zc.update(_dets([[80, 80, 120, 120], [70, 70, 210, 210]], [2, 7], tid=[1, 2],
                    reclassified=mark), now=t)
    return zc, zc.update(_dets([[120, 120, 160, 160], [110, 110, 250, 250]], [2, 7],
                               tid=[1, 2], reclassified=mark), now=t + 1)


def test_reclassified_small_truck_not_suppressed_as_cabin():
    """A size-demoted truck (class 2, flagged) overlapping a real truck box must
    still count: the cabin filter only applies to genuine cars."""
    zc, ev = _cabin_frame(flagged=True)
    assert zc.suppressed_cabin == 0, zc.suppressed_cabin
    assert any(e["track_id"] == 1 for e in ev), f"reclassified truck dropped: {ev}"


def test_genuine_car_cabin_still_suppressed():
    """Unflagged car inside a truck box is still discarded as the cabin."""
    zc, ev = _cabin_frame(flagged=None)
    assert zc.suppressed_cabin == 1, zc.suppressed_cabin
    assert not any(e["track_id"] == 1 for e in ev), ev


def test_latch_survives_exit_and_reset_clears_it():
    """One vehicle inside both overlapping zones counts once per zone; the
    latch survives an exit + re-entry; reset clears latches; empty dets safe."""
    za = sv.PolygonZone(polygon=np.array([(100, 200), (350, 200), (350, 500), (100, 500)]))
    zb = sv.PolygonZone(polygon=np.array([(340, 200), (600, 200), (600, 500), (340, 500)]))
    zc = ZoneCounter([za, zb], class_labels={3: "motor"}, dedup_cooldown_sec=0.0,
                     min_displacement_px=0, min_track_frames=1)
    t = time.monotonic()
    zc.update(_dets([[300, 300, 400, 400]], [3], tid=[1]), now=t)   # centroid in both
    assert zc.total == 2, zc.total
    zc.update(_dets([[20, 250, 80, 350]], [3], tid=[1]), now=t + 1)  # exits
    zc.update(_dets([[300, 300, 400, 400]], [3], tid=[1]), now=t + 2)  # re-enters
    assert zc.total == 2, zc.total
    zc.update(sv.Detections.empty(), now=t + 3)
    assert zc.total == 2
    zc.reset()
    assert zc.total == 0
    zc.update(_dets([[300, 300, 400, 400]], [3], tid=[1]), now=t + 4)
    assert zc.total == 2, "reset must clear per-zone latches"


def test_audit_logger_writes_header_and_row():
    """One accepted count -> header + one CSV row."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [200, 0], [200, 200], [0, 200]]))
    zc = ZoneCounter([zone], class_labels={3: "motor"}, min_displacement_px=0, min_track_frames=1)
    ev = zc.update(_dets([[90, 90, 110, 110]], [3], tid=[7]), now=time.monotonic())
    tmp = os.path.join(tempfile.mkdtemp(), "count_audit.csv")
    al = AuditLogger(tmp)
    al.write(ev[0])
    al.close()
    rows = list(csv.reader(open(tmp)))
    assert rows[0] == ["timestamp", "track_id", "class_name", "confidence",
                       "centroid_x", "centroid_y"], rows[0]
    assert len(rows) == 2 and rows[1][1] == "7" and rows[1][2] == "motor", rows


# ---------------------------------------------------------------------------
# RT-DETR anomaly defenses: cross-class dedup, pickup re-attribution,
# motorcycle turn inertia, two-tier area demotion.
# ---------------------------------------------------------------------------

def test_cross_class_queries_deduped_before_tracking():
    """Two RT-DETR queries on one vehicle (truck + car, same box) collapse to
    a single detection so ByteTrack cannot spawn a second track."""
    d = _dets([[0, 0, 100, 100], [2, 2, 102, 102], [300, 300, 340, 340]],
              [7, 2, 3])
    kept = cross_class_nms(d, duplicate_iou=0.5)
    assert len(kept) == 2, kept.class_id
    assert 3 in list(kept.class_id)                       # far bike untouched
    assert sum(1 for c in kept.class_id if c in (2, 7)) == 1, kept.class_id


def test_pickup_mid_zone_flip_single_count():
    """One tracker_id enters as truck, flips to car inside the zone: the count
    must re-attribute truck -> car atomically, never double count."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [400, 0], [400, 400], [0, 400]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 7: "truk/bus"},
                     min_displacement_px=0)
    t = time.monotonic()
    evs = []
    for i in range(2):   # truck on the boundary
        evs += zc.update(_dets([[90, 90, 130, 130]], [7], tid=[1]), now=t + i)
    assert zc.total == 1 and zc.total_by_class["truk/bus"] == 1, dict(zc.total_by_class)
    for i in range(5):   # perspective shift: RT-DETR now reads car mid-zone
        evs += zc.update(_dets([[92, 92, 132, 132]], [2], tid=[1]),
                         now=t + 2 + i)
    assert zc.total == 1, zc.total
    assert zc.total_by_class["truk/bus"] == 0, dict(zc.total_by_class)
    assert zc.total_by_class["mobil"] == 1, dict(zc.total_by_class)
    re = [e for e in evs if e.get("reattributed_from")]
    assert len(re) == 1 and re[0]["class_name"] == "mobil", evs
    assert re[0]["reattributed_from"] == "truk/bus", re


def test_pickup_flip_back_to_truck_is_noop():
    """A one-frame car flicker that votes back to truck must not move buckets."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [400, 0], [400, 400], [0, 400]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 7: "truk/bus"},
                     min_displacement_px=0)
    t = time.monotonic()
    for i in range(3):
        zc.update(_dets([[90, 90, 130, 130]], [7], tid=[1]), now=t + i)
    ev = zc.update(_dets([[92, 92, 132, 132]], [2], tid=[1]), now=t + 3)
    assert not ev, ev
    assert zc.total_by_class["truk/bus"] == 1, dict(zc.total_by_class)
    assert zc.total_by_class["mobil"] == 0, dict(zc.total_by_class)


def test_turning_motorcycle_inertia():
    """Motorcycle for 4 frames, then car flicker for 3 while turning inside
    the zone: inertia holds motorcycle, no car count is ever registered."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [400, 0], [400, 400], [0, 400]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 3: "motor"},
                     min_displacement_px=0)
    t = time.monotonic()
    for i in range(4):
        zc.update(_dets([[90, 90, 130, 130]], [3], tid=[1]), now=t + i)
    evs = []
    for i in range(3):
        evs += zc.update(_dets([[88, 88, 148, 148]], [2], tid=[1]),
                         now=t + 4 + i)
    assert zc.total == 1, zc.total
    assert zc.total_by_class["motor"] == 1, dict(zc.total_by_class)
    assert zc.total_by_class.get("mobil", 0) == 0, dict(zc.total_by_class)
    assert not any(e.get("reattributed_from") for e in evs), evs


def test_motorcycle_inertia_beats_conf_tiebreak():
    """Adversarial flicker: 4 motorcycle frames vs 4 high-confidence car
    frames (count tie, car conf wins the vote). Inertia must hold motor."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [400, 0], [400, 400], [0, 400]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 3: "motor"},
                     min_displacement_px=0)
    t = time.monotonic()
    for i in range(4):
        zc.update(_dets([[90, 90, 130, 130]], [3], conf=[0.5], tid=[1]),
                  now=t + i)
    evs = []
    for i in range(4):
        evs += zc.update(_dets([[88, 88, 148, 148]], [2], conf=[0.95], tid=[1]),
                         now=t + 4 + i)
    assert zc.total == 1, zc.total
    assert zc.total_by_class["motor"] == 1, dict(zc.total_by_class)
    assert zc.total_by_class.get("mobil", 0) == 0, dict(zc.total_by_class)
    assert not any(e.get("reattributed_from") for e in evs), evs


def test_motorcycle_inertia_yields_to_long_car_run():
    """A sustained car run (> motor_flip_frames) is a real class transition,
    not turn flicker: re-attribution moves the count to car."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [400, 0], [400, 400], [0, 400]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 3: "motor"},
                     min_displacement_px=0)
    t = time.monotonic()
    for i in range(4):
        zc.update(_dets([[90, 90, 130, 130]], [3], tid=[1]), now=t + i)
    evs = []
    for i in range(6):   # 6 > motor_flip_frames
        evs += zc.update(_dets([[90, 90, 140, 140]], [2], tid=[1]),
                         now=t + 4 + i)
    assert zc.total == 1, zc.total
    assert zc.total_by_class.get("motor", 0) == 0, dict(zc.total_by_class)
    assert zc.total_by_class["mobil"] == 1, dict(zc.total_by_class)
    assert any(e.get("reattributed_from") == "motor" for e in evs), evs


def test_approaching_car_does_not_upgrade_to_truck():
    """Car for 4 frames entering the zone, then truck/bus for 5 frames as it
    approaches (box inflates with perspective): the car lock holds the class,
    no flip and no re-attribution to truck/bus."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [400, 0], [400, 400], [0, 400]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 7: "truk/bus"},
                     min_displacement_px=0)
    t = time.monotonic()
    evs = []
    for i in range(4):
        evs += zc.update(_dets([[90, 90, 130, 130]], [2], tid=[1]), now=t + i)
    for i in range(5):
        box = [[90 - i * 2, 90 - i * 2, 130 + i * 6, 130 + i * 6]]  # grows near cam
        evs += zc.update(_dets(box, [7], tid=[1]), now=t + 4 + i)
    assert zc.total == 1, zc.total
    assert zc.total_by_class["mobil"] == 1, dict(zc.total_by_class)
    assert zc.total_by_class.get("truk/bus", 0) == 0, dict(zc.total_by_class)
    assert not any(e.get("reattributed_from") for e in evs), evs


def test_turning_car_box_inflation_stays_car():
    """Same lock from a diagonal turn: box area balloons past truck_min_area
    while the tracker id stays a car."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [400, 0], [400, 400], [0, 400]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 5: "truk/bus"},
                     min_displacement_px=0)
    t = time.monotonic()
    for i in range(3):
        zc.update(_dets([[100, 100, 140, 140]], [2], tid=[1]), now=t + i)
    for i in range(4):
        zc.update(_dets([[100, 100, 200, 200]], [5], tid=[1]), now=t + 3 + i)
    assert zc.total == 1, zc.total
    assert zc.total_by_class["mobil"] == 1, dict(zc.total_by_class)
    assert zc.total_by_class.get("truk/bus", 0) == 0, dict(zc.total_by_class)


def test_truck_opening_still_reattributes_to_car():
    """Car lock must not block the pickup path: a track that opens as truck
    still re-attributes to car once the vote stabilizes."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [400, 0], [400, 400], [0, 400]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 7: "truk/bus"},
                     min_displacement_px=0)
    t = time.monotonic()
    evs = []
    for i in range(2):
        evs += zc.update(_dets([[90, 90, 130, 130]], [7], tid=[1]), now=t + i)
    for i in range(4):
        evs += zc.update(_dets([[92, 92, 132, 132]], [2], tid=[1]), now=t + 2 + i)
    assert zc.total == 1, zc.total
    assert zc.total_by_class["mobil"] == 1, dict(zc.total_by_class)
    assert zc.total_by_class.get("truk/bus", 0) == 0, dict(zc.total_by_class)
    assert any(e.get("reattributed_from") == "truk/bus" for e in evs), evs


def test_small_car_demoted_to_motorcycle():
    """Tiny car box (turning bike) -> motorcycle; normal car/truck untouched."""
    d = _dets([[0, 0, 50, 50], [0, 100, 200, 300], [0, 400, 60, 460]],
              [2, 2, 7])
    out = reclassify_small_cars(d, car_min_area=3000)   # 2500 < 3000
    assert list(out.class_id) == [3, 2, 7], out.class_id


def test_size_heuristics_order_truck_to_car_not_motorcycle():
    """Truck demoted to car must land on car even if its area is also under
    car_min_area: the car sweep runs before the truck sweep."""
    d = _dets([[0, 0, 50, 50]], [7])   # 2500 < both thresholds
    out = apply_size_heuristics(d, truck_min_area=8000, car_min_area=3000)
    assert list(out.class_id) == [2], out.class_id


def test_small_car_demotion_disabled_and_empty():
    d = _dets([[0, 0, 10, 10]], [2])
    assert list(reclassify_small_cars(d, car_min_area=0).class_id) == [2]
    d2 = _dets([[0, 0, 100, 100]], [2])   # area == threshold is NOT below
    assert list(reclassify_small_cars(d2, car_min_area=10000).class_id) == [2]
    empty = sv.Detections.empty()
    assert reclassify_small_cars(empty, car_min_area=3000) is empty


def test_small_bus_demoted_to_car():
    """Bus (5) below truck_min_area is a pickup/box car -> car, not motor."""
    d = _dets([[0, 0, 60, 60], [0, 100, 300, 300]], [5, 5])
    out = reclassify_small_trucks(d, truck_min_area=8000)
    assert list(out.class_id) == [2, 5], out.class_id


def test_demoted_car_still_suppressed_as_cabin():
    """A car box demoted to motorcycle by car_min_area, but geometrically the
    cabin of a truck, must still be discarded — not counted as a bike."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [400, 0], [400, 400], [0, 400]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 3: "motor", 7: "truk/bus"},
                     min_displacement_px=50)
    t = time.monotonic()
    # cabin box 40x40=1600 (< car_min_area) + truck box; both move 50px.
    # Detector has already relabeled the cabin to motor (class 3) and set the
    # demoted_car tag.
    zc.update(_dets([[80, 80, 120, 120], [70, 70, 210, 210]], [3, 7], tid=[1, 2],
                    demoted_car=[True, False]), now=t)
    ev = zc.update(_dets([[120, 120, 160, 160], [110, 110, 250, 250]], [3, 7],
                         tid=[1, 2], demoted_car=[True, False]), now=t + 1)
    assert zc.suppressed_cabin == 1, zc.suppressed_cabin
    assert not any(e["track_id"] == 1 for e in ev), ev
    assert any(e["track_id"] == 2 for e in ev), ev


def test_demoted_car_outside_truck_counts_as_motor():
    """Same demoted-car tag but no truck box overhead: the turning bike counts
    as a motorcycle, not suppressed."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [400, 0], [400, 400], [0, 400]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 3: "motor"},
                     min_displacement_px=0, min_track_frames=1)
    t = time.monotonic()
    ev = zc.update(_dets([[90, 90, 130, 130]], [3], tid=[1],
                         demoted_car=[True]), now=t)
    assert len(ev) == 1 and ev[0]["class_name"] == "motor", ev


def test_fast_vehicle_pre_zone_displacement_counted_in_single_zone_frame():
    """The displacement anchor is taken on the first frame the id appears
    ANYWHERE in the view, not on zone entry: a fast bike tracked for two frames
    above the polygon already sits 120px from its anchor on the single frame it
    lands inside, so it counts. A pot jittering on the boundary never does."""
    zone = sv.PolygonZone(polygon=np.array([[100, 200], [600, 200],
                                            [600, 500], [100, 500]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 3: "motor"},
                     min_displacement_px=30)
    t = time.monotonic()
    assert not zc.update(_dets([[290, 110, 310, 130]], [3], tid=[1]), now=t)
    assert not zc.update(_dets([[290, 150, 310, 170]], [3], tid=[1]), now=t + 1)
    ev = zc.update(_dets([[290, 230, 310, 250]], [3], tid=[1]), now=t + 2)
    assert len(ev) == 1 and ev[0]["class_name"] == "motor", ev
    assert zc.total == 1 and zc.total_by_class["motor"] == 1
    assert not zc.update(sv.Detections.empty(), now=t + 3)   # latch holds
    assert zc.total == 1

    # static roadside object on the boundary: jitter stays far below 30px
    pot = ZoneCounter([zone], class_labels={2: "mobil", 3: "motor"},
                      min_displacement_px=30)
    for k, dy in enumerate((0, 3, -2, 4, -3, 2)):
        assert not pot.update(_dets([[590, 240 + dy, 610, 260 + dy]], [2], tid=[9]),
                              now=t + k)
    assert pot.total == 0


def test_in_zone_occlusion_id_switch_not_double_counted():
    """tid=1 is counted inside the zone, occluded for 3 frames, then respawns
    25px away as tid=2 and drives another 40px deeper. Ghost handover proves it
    is the same vehicle: total stays 1 and the latch moves to tid=2."""
    zone = sv.PolygonZone(polygon=np.array([[100, 200], [600, 200],
                                            [600, 500], [100, 500]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 3: "motor"},
                     min_displacement_px=30)
    t = time.monotonic()
    assert not zc.update(_dets([[290, 240, 310, 260]], [2], tid=[1]), now=t)
    ev = zc.update(_dets([[290, 280, 310, 300]], [2], tid=[1]), now=t + 1)
    assert len(ev) == 1 and zc.total == 1, ev
    assert not zc.update(_dets([[290, 320, 310, 340]], [2], tid=[1]), now=t + 2)
    for k in (3, 4, 5):                                  # 3 frames occluded
        assert not zc.update(sv.Detections.empty(), now=t + k)
    # respawn 25px from the ghost: born f6, ghost last seen f3 -> gap 3
    assert not zc.update(_dets([[310, 335, 330, 355]], [2], tid=[2]), now=t + 6)
    ev = zc.update(_dets([[310, 375, 330, 395]], [2], tid=[2]), now=t + 7)
    assert not ev and zc.total == 1, ev                  # handover, not a count
    assert zc.ghost_handovers == 1
    assert zc.total_by_class["mobil"] == 1
    assert zc.counted_id_class[0][2] == "mobil" and 2 in zc._counted[0]
    assert 1 not in zc._zone_ghost[0]                    # ghost retired


def test_in_zone_ghost_handover_preserves_reattribution():
    """A pickup counted as truck/bus is occluded for 2 frames and comes back as
    tid=2 reading car. The handover carries the counted latch AND the class
    history over, so atomic re-attribution still fires: total stays 1 while the
    count moves from truk/bus to mobil."""
    zone = sv.PolygonZone(polygon=np.array([[100, 200], [600, 200],
                                            [600, 500], [100, 500]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 3: "motor",
                                           5: "truk/bus", 7: "truk/bus"},
                     min_displacement_px=0, min_track_frames=1)
    t = time.monotonic()
    ev = zc.update(_dets([[280, 230, 320, 270]], [7], tid=[1]), now=t)
    assert len(ev) == 1 and ev[0]["class_name"] == "truk/bus", ev
    assert not zc.update(_dets([[280, 270, 320, 310]], [7], tid=[1]), now=t + 1)
    for k in (2, 3):                                     # 2 frames occluded
        assert not zc.update(sv.Detections.empty(), now=t + k)
    later = []
    for k, y in ((4, 280), (5, 320), (6, 360)):          # stabilizes as car
        later += zc.update(_dets([[290, y, 330, y + 40]], [2],
                                 conf=[0.95], tid=[2]), now=t + k)
    assert zc.total == 1, later                          # never counted twice
    assert zc.ghost_handovers == 1
    assert zc.total_by_class["truk/bus"] == 0
    assert zc.total_by_class["mobil"] == 1
    flips = [e for e in later if e.get("reattributed_from")]
    assert len(flips) == 1, flips
    assert flips[0]["reattributed_from"] == "truk/bus"
    assert flips[0]["class_name"] == "mobil" and flips[0]["track_id"] == 2


def test_parallel_coexisting_vehicles_both_counted():
    """Two bikes share frames 1-4, so tid=2 was visible while tid=1 was alive.
    tid=1 is counted, arms a ghost, then vanishes; tid=2 rides in 27px from that
    ghost. The non-coexistence invariant refuses the handover and tid=2 counts:
    side-by-side riders never steal each other's latch."""
    zone = sv.PolygonZone(polygon=np.array([[100, 200], [600, 200],
                                            [600, 500], [100, 500]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 3: "motor"},
                     min_displacement_px=30)
    t = time.monotonic()
    both = [([[290, 110, 310, 130], [330, 110, 350, 130]], [1, 2], t),
            ([[290, 150, 310, 170], [333, 115, 353, 135]], [1, 2], t + 1)]
    for xyxy, tid, now in both:                          # both born frame 1
        assert not zc.update(_dets(xyxy, [3, 3], tid=tid), now=now)
    ev = zc.update(_dets([[290, 220, 310, 240], [328, 112, 348, 132]],
                         [3, 3], tid=[1, 2]), now=t + 2)  # tid=1 lands inside
    assert len(ev) == 1 and ev[0]["track_id"] == 1, ev
    ev = zc.update(_dets([[290, 260, 310, 280], [330, 120, 350, 140]],
                         [3, 3], tid=[1, 2]), now=t + 3)  # ghost armed at f4
    assert not ev and zc.total == 1
    for k, xy in ((4, [332, 116, 352, 136]), (5, [328, 118, 348, 138])):
        assert not zc.update(_dets([xy], [3], tid=[2]), now=t + k)  # tid=1 gone
    ev = zc.update(_dets([[300, 235, 320, 255]], [3], tid=[2]), now=t + 6)
    assert len(ev) == 1 and ev[0]["track_id"] == 2, ev
    assert zc.total == 2 and zc.total_by_class["motor"] == 2
    assert zc.ghost_handovers == 0
    assert 1 in zc._zone_ghost[0]        # ghost was available and refused
    assert {1, 2} <= zc._counted[0]


def test_stale_tracker_id_pruning_preserves_counts_and_ghost_handover():
    """Stale track state is pruned every 100 frames without altering total
    counts or breaking active Ghost ID Handover, and max_stale_frames > ghost_ttl_frames."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [400, 0], [400, 400], [0, 400]]))
    clamped = ZoneCounter([zone], ghost_ttl_frames=20, max_stale_frames=10)
    assert clamped.max_stale_frames > clamped.ghost_ttl_frames

    zc = ZoneCounter([zone], class_labels={2: "mobil", 3: "motor"},
                     ghost_ttl_frames=10, max_stale_frames=150,
                     min_displacement_px=0, min_track_frames=1, debug=True)
    t = time.monotonic()

    # Frame 1: tid=1 counted inside zone
    ev = zc.update(_dets([[90, 90, 130, 130]], [2], tid=[1]), now=t)
    assert len(ev) == 1 and zc.total == 1 and zc.total_by_class["mobil"] == 1

    # Frames 2..100: empty frames; at frame 100 tid=1 is < max_stale_frames so retained
    for k in range(2, 101):
        zc.update(sv.Detections.empty(), now=t + k)
    assert zc._frame_idx == 100
    for store in (zc._start, zc.class_hist, zc._first_cls, zc._last_seen_frame,
                  zc._born_frame, zc._birth_centroid, zc._last_centroid,
                  zc._prev, zc._age, zc.trails, zc._counted[0], zc.counted_id_class[0]):
        assert 1 in store

    # Frames 101..195: empty frames
    for k in range(101, 196):
        zc.update(sv.Detections.empty(), now=t + k)

    # Frames 196..197: active tid=2 enters and arms ghost inside zone
    ev = zc.update(_dets([[200, 200, 240, 240]], [2], tid=[2]), now=t + 196)
    assert len(ev) == 1 and zc.total == 2 and zc.total_by_class["mobil"] == 2
    zc.update(_dets([[200, 200, 240, 240]], [2], tid=[2]), now=t + 197)

    # Frames 198..200: empty frames; frame 200 prunes tid=1 (last_seen=1 < 200-150)
    for k in range(198, 201):
        zc.update(sv.Detections.empty(), now=t + k)
    assert zc._frame_idx == 200
    for store in (zc._start, zc.class_hist, zc._first_cls, zc._last_seen_frame,
                  zc._born_frame, zc._birth_centroid, zc._last_centroid,
                  zc._prev, zc._age, zc._missing, zc._zone_still, zc.trails,
                  zc._counted[0], zc.counted_id_class[0], zc._zone_ghost[0]):
        assert 1 not in store
    assert zc.total == 2 and zc.total_by_class["mobil"] == 2
    assert 2 in zc._last_seen_frame and 2 in zc._zone_ghost[0]

    # Frame 201: tid=3 respawns near tid=2's ghost -> Ghost ID Handover without double count
    ev = zc.update(_dets([[210, 210, 250, 250]], [2], tid=[3]), now=t + 201)
    assert not ev
    assert zc.ghost_handovers == 1
    assert zc.total == 2 and zc.total_by_class["mobil"] == 2
    assert 3 in zc._counted[0] and zc.counted_id_class[0][3] == "mobil"


def test_flow_gate_default_none_counts_all_directions():
    """allowed_directions=None preserves legacy counting for every direction."""
    zone = sv.PolygonZone(polygon=np.array([[100, 100], [500, 100], [500, 500], [100, 500]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil"}, min_displacement_px=30,
                     dedup_cooldown_sec=0.0, allowed_directions=None)
    t = time.monotonic()
    moves = [
        (1, (300, 50), (300, 200)),   # down
        (2, (300, 550), (300, 400)),  # up
        (3, (550, 300), (400, 300)),  # left
        (4, (50, 300), (200, 300)),   # right
    ]
    for idx, (tid, (x0, y0), (x1, y1)) in enumerate(moves):
        zc.update(_dets([[x0 - 10, y0 - 10, x0 + 10, y0 + 10]], [2], tid=[tid]), now=t + idx * 2)
        ev = zc.update(_dets([[x1 - 10, y1 - 10, x1 + 10, y1 + 10]], [2], tid=[tid]), now=t + idx * 2 + 1)
        assert len(ev) == 1
    assert zc.total == 4
    assert zc.suppressed_direction == 0


def test_flow_gate_bidirectional_underpass():
    """Underpass ['up', 'down'] counts both vertical flows and suppresses sideways grazers."""
    zone = sv.PolygonZone(polygon=np.array([[100, 100], [500, 100], [500, 500], [100, 500]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 3: "motor"}, min_displacement_px=30,
                     dedup_cooldown_sec=0.0, allowed_directions=["up", "down"])
    t = time.monotonic()

    # Approaching (down)
    zc.update(_dets([[240, 40, 260, 60]], [2], tid=[1]), now=t)
    ev1 = zc.update(_dets([[240, 190, 260, 210]], [2], tid=[1]), now=t + 1)
    # Receding (up)
    zc.update(_dets([[340, 540, 360, 560]], [3], tid=[2]), now=t + 2)
    ev2 = zc.update(_dets([[340, 390, 360, 410]], [3], tid=[2]), now=t + 3)
    # Sideways left & right edge-grazers -> suppressed
    zc.update(_dets([[540, 290, 560, 310]], [2], tid=[3]), now=t + 4)
    ev3 = zc.update(_dets([[390, 290, 410, 310]], [2], tid=[3]), now=t + 5)
    zc.update(_dets([[40, 240, 60, 260]], [2], tid=[4]), now=t + 6)
    ev4 = zc.update(_dets([[190, 240, 210, 260]], [2], tid=[4]), now=t + 7)

    assert len(ev1) == 1 and len(ev2) == 1
    assert not ev3 and not ev4
    assert zc.total == 2
    assert zc.suppressed_direction == 2


def test_flow_gate_combination_turns_and_per_zone_override():
    """['down', 'left'] @ 75deg accepts down and down-left turns, rejects up; supports vectors and zone overrides."""
    zone = sv.PolygonZone(polygon=np.array([[100, 100], [500, 100], [500, 500], [100, 500]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil"}, min_displacement_px=30,
                     dedup_cooldown_sec=0.0, allowed_directions=["down", "left"],
                     flow_max_angle_deg=75.0)
    t = time.monotonic()

    # Straight down
    zc.update(_dets([[290, 40, 310, 60]], [2], tid=[1]), now=t)
    ev_down = zc.update(_dets([[290, 190, 310, 210]], [2], tid=[1]), now=t + 1)
    # Down-left diagonal turn
    zc.update(_dets([[440, 140, 460, 160]], [2], tid=[2]), now=t + 2)
    ev_turn = zc.update(_dets([[340, 240, 360, 260]], [2], tid=[2]), now=t + 3)
    # Opposite (up) -> rejected
    zc.update(_dets([[290, 540, 310, 560]], [2], tid=[3]), now=t + 4)
    ev_up = zc.update(_dets([[290, 390, 310, 410]], [2], tid=[3]), now=t + 5)

    assert len(ev_down) == 1 and len(ev_turn) == 1 and not ev_up
    assert zc.total == 2 and zc.suppressed_direction == 1

    # Numeric vector + zero-length vector ignored + per-zone override
    z_dicts = [
        {"polygon": [[100, 100], [300, 100], [300, 300], [100, 300]],
         "allowed_directions": ["down-right", [0.0, 0.0]]},
        {"polygon": [[320, 100], [520, 100], [520, 300], [320, 300]],
         "allowed_directions": [[0.0, -5.0]]},
    ]
    zc2 = ZoneCounter(z_dicts, class_labels={2: "mobil"}, min_displacement_px=30,
                      dedup_cooldown_sec=0.0, allowed_directions=["up"],
                      flow_max_angle_deg=45.0)
    # Moving down-right in zone 0 passes zone 0 override despite global ["up"]
    zc2.update(_dets([[90, 90, 110, 110]], [2], tid=[10]), now=t)
    ev_z0 = zc2.update(_dets([[190, 190, 210, 210]], [2], tid=[10]), now=t + 1)
    # Moving down in zone 1 (which allows only [0, -5] == up) is rejected
    zc2.update(_dets([[410, 40, 430, 60]], [2], tid=[11]), now=t + 2)
    ev_z1 = zc2.update(_dets([[410, 190, 430, 210]], [2], tid=[11]), now=t + 3)
    assert len(ev_z0) == 1 and not ev_z1
    assert zc2.total == 1 and zc2.suppressed_direction == 1


def test_flow_gate_ghost_handover_preserves_start_anchor():
    """Ghost ID Handover inherits _start so respawned tracks keep true flow direction."""
    zone = sv.PolygonZone(polygon=np.array([[100, 200], [600, 200], [600, 500], [100, 500]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil"}, min_displacement_px=30,
                     dedup_cooldown_sec=4.0, allowed_directions=["down"],
                     flow_max_angle_deg=75.0)
    t = time.monotonic()
    zc.update(_dets([[290, 90, 310, 110]], [2], tid=[1]), now=t)     # _start=(300, 100)
    ev = zc.update(_dets([[290, 250, 310, 270]], [2], tid=[1]), now=t + 1)  # down -> counted
    assert len(ev) == 1 and zc.total == 1
    zc.update(_dets([[290, 290, 310, 310]], [2], tid=[1]), now=t + 2)  # ghost at (300, 300)
    for k in (3, 4, 5):
        zc.update(sv.Detections.empty(), now=t + k)
    # Respawn near ghost with slight lateral offset; inherited _start=(300,100) gives dy=+215 (down)
    ev_ho = zc.update(_dets([[300, 305, 320, 325]], [2], tid=[2]), now=t + 6)
    ev_ho += zc.update(_dets([[300, 325, 320, 345]], [2], tid=[2]), now=t + 7)
    assert not ev_ho
    assert zc.ghost_handovers == 1
    assert zc.total == 1
    assert zc.suppressed_direction == 0
    assert 2 in zc._counted[0]


def test_bottom_center_polygon_trigger_counts_edge_vehicle():
    """Vehicle whose centroid is outside the polygon top edge but whose
    bottom-center (cx, y2) lands inside is counted immediately."""
    zone = sv.PolygonZone(polygon=np.array([[100, 200], [600, 200], [600, 500], [100, 500]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil"})
    t = time.monotonic()
    # Frame 1: centroid (300, 150), bottom_center (300, 170) -> both above y=200
    assert not zc.update(_dets([[280, 130, 320, 170]], [2], tid=[1]), now=t)
    # Frame 2: centroid (300, 180) still < 200, but bottom_center (300, 210) >= 200; disp=30px >= 12px
    ev = zc.update(_dets([[280, 150, 320, 210]], [2], tid=[1]), now=t + 1)
    assert len(ev) == 1 and ev[0]["class_name"] == "mobil"
    assert zc.total == 1


def test_dense_following_motorcycles_not_swallowed_by_ghost_handover():
    """Closely following motorcycles (~45px apart) in dense traffic both count,
    and motor vs non-motor class mismatch never hands over."""
    zone = sv.PolygonZone(polygon=np.array([[100, 200], [600, 200], [600, 500], [100, 500]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 3: "motor"}, dedup_cooldown_sec=0.0)
    t = time.monotonic()
    # Bike 1 enters, moves 20px (>= 12px default), counts, arms ghost at (300, 270), then vanishes
    zc.update(_dets([[290, 240, 310, 260]], [3], tid=[1]), now=t)
    ev1 = zc.update(_dets([[290, 260, 310, 280]], [3], tid=[1]), now=t + 1)
    assert len(ev1) == 1 and zc.total == 1
    # Bike 2 spawns 45px behind ghost at (300, 225) on next frame and moves 20px
    zc.update(_dets([[290, 215, 310, 235]], [3], tid=[2]), now=t + 2)
    ev2 = zc.update(_dets([[290, 235, 310, 255]], [3], tid=[2]), now=t + 3)
    assert len(ev2) == 1
    assert zc.total == 2 and zc.total_by_class["motor"] == 2
    assert zc.ghost_handovers == 0

    # Class mismatch guard: a car ghost at (450, 300) cannot be claimed by a motorcycle born 10px away
    zc.update(_dets([[430, 260, 470, 300]], [2], tid=[10]), now=t + 4)
    ev_car = zc.update(_dets([[430, 280, 470, 320]], [2], tid=[10]), now=t + 5)
    assert len(ev_car) == 1 and zc.total_by_class["mobil"] == 1
    zc.update(_dets([[445, 295, 465, 315]], [3], tid=[11]), now=t + 6)
    ev_moto = zc.update(_dets([[445, 315, 465, 335]], [3], tid=[11]), now=t + 7)
    assert len(ev_moto) == 1 and ev_moto[0]["class_name"] == "motor"
    assert zc.ghost_handovers == 0
    assert zc.total_by_class["motor"] == 3 and zc.total_by_class["mobil"] == 1


def test_mobil_strict_lock_prevents_late_upgrade_or_reattribution_to_truk():
    """A track first seen or counted as mobil never upgrades or reattributes to
    truk/bus even after 15+ consecutive inflated truck predictions near the bottom."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [500, 0], [500, 500], [0, 500]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 5: "truk/bus", 7: "truk/bus"})
    t = time.monotonic()

    # Case 1: Counted as mobil after 2 frames, then 16 frames of raw class 7 (box inflation)
    zc.update(_dets([[100, 100, 140, 140]], [2], tid=[1]), now=t)
    ev = zc.update(_dets([[100, 120, 145, 165]], [2], tid=[1]), now=t + 1)
    assert len(ev) == 1 and ev[0]["class_name"] == "mobil"
    later = []
    for k in range(16):
        later += zc.update(_dets([[90, 140 + k * 10, 210, 260 + k * 10]], [7],
                                 conf=[0.95], tid=[1]), now=t + 2 + k)
    assert not any(e.get("reattributed_from") for e in later), later
    assert zc.total == 1
    assert zc.total_by_class["mobil"] == 1
    assert zc.total_by_class.get("truk/bus", 0) == 0

    # Case 2: Seen once as mobil before zone, then 10 frames of raw class 7 entering zone
    zc.update(_dets([[300, 10, 340, 50]], [2], tid=[2]), now=t + 20)
    ev2 = []
    for k in range(10):
        ev2 += zc.update(_dets([[300, 30 + k * 15, 410, 130 + k * 15]], [7],
                               conf=[0.95], tid=[2]), now=t + 21 + k)
    assert len(ev2) == 1 and ev2[0]["class_name"] == "mobil"
    assert zc.total == 2
    assert zc.total_by_class["mobil"] == 2
    assert zc.total_by_class.get("truk/bus", 0) == 0


def test_receding_medium_truck_locked_against_shrink_demotion():
    """Medium truck confirmed near/mid as truk/bus moves up (dy < 0) into upper
    polygon where raw class drops to mobil while remaining >= 0.85x effective
    truck area: stays truk/bus via _truck_lock_latched (and across Ghost ID Handover)."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [500, 0], [500, 500], [0, 500]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 5: "truk/bus", 7: "truk/bus"},
                     dedup_cooldown_sec=0.0)
    t = time.monotonic()

    # Frames 1..2: fast-track large/tall medium truck entering near bottom/mid (dy < 0)
    # y2=440 -> eff_area ~ 20640 -> 1.5x = 30960, eff_h ~ 87 -> 1.2x = 104
    for k, box in enumerate(([160, 260, 340, 440], [165, 210, 345, 390])):
        d = _dets([box], [7], tid=[1])
        d.data["perspective_truck"] = np.array([True], dtype=bool)
        ev_k = zc.update(d, now=t + k)
        if k == 1:
            # Zone entry never waits for confirmation: counted immediately once disp >= 12px
            assert len(ev_k) == 1 and ev_k[0]["class_name"] == "truk/bus"
    assert 1 in zc._truck_lock_latched

    # Frames 3..8: moves away upward into upper polygon; raw class drops to mobil (2)
    # while perspective area stays >= 0.85 * effective_truck_min_area(y2)
    later = []
    for k in range(6):
        y2 = 240 - k * 25
        w, h = 110 - k * 4, 100 - k * 4  # 11000..7200 px2 (> 0.85x eff_truck_min_area)
        later += zc.update(_dets([[250 - w / 2, y2 - h, 250 + w / 2, y2]],
                                 [2], conf=[0.95], tid=[1]), now=t + 2 + k)
    assert not any(e.get("reattributed_from") for e in later), later
    assert zc.total == 1
    assert zc.total_by_class["truk/bus"] == 1
    assert zc.total_by_class.get("mobil", 0) == 0

    # Brief occlusion + Ghost ID Handover preserves _truck_lock_latched on tid=2
    zc.update(sv.Detections.empty(), now=t + 10)
    ev_ho = zc.update(_dets([[205, 45, 295, 115]], [2], conf=[0.95], tid=[2]), now=t + 11)
    ev_ho += zc.update(_dets([[205, 35, 295, 105]], [2], conf=[0.95], tid=[2]), now=t + 12)
    assert not ev_ho and zc.ghost_handovers == 1
    assert 2 in zc._truck_lock_latched
    assert zc.total == 1 and zc.total_by_class["truk/bus"] == 1


def test_approaching_medium_truck_promotes_after_consecutive_validated_frames():
    """Track starting as mobil resists 3-4 glare TRUCK_CLS frames (< truck_confirm_frames=5),
    promotes on 5th consecutive perspective-validated frame, or fast-tracks in 2 frames
    when >= 1.5x effective area and >= 1.2x effective height."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [600, 0], [600, 600], [0, 600]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 5: "truk/bus", 7: "truk/bus"},
                     truck_min_area=12000.0)
    t = time.monotonic()

    # Case A: Rain/night glare protection — 4 consecutive TRUCK_CLS frames (<5) stay mobil; 5th promotes
    zc.update(_dets([[270, 80, 330, 140]], [2], tid=[1]), now=t)
    ev = zc.update(_dets([[268, 100, 332, 165]], [2], tid=[1]), now=t + 1)
    assert len(ev) == 1 and ev[0]["class_name"] == "mobil"
    assert 1 in zc._car_lock_latched

    for step in range(4):
        d = _dets([[250, 150 + step * 20, 350, 260 + step * 20]], [7], conf=[0.85], tid=[1])  # ~11000 px2 (< 1.5x eff)
        d.data["perspective_truck"] = np.array([True], dtype=bool)
        assert not zc.update(d, now=t + 2 + step)
    assert 1 in zc._car_lock_latched and 1 not in zc._truck_lock_latched
    assert zc.total_by_class["mobil"] == 1

    d5 = _dets([[240, 230, 360, 350]], [7], conf=[0.85], tid=[1])
    d5.data["perspective_truck"] = np.array([True], dtype=bool)
    ev5 = zc.update(d5, now=t + 6)
    assert len(ev5) == 1 and ev5[0]["reattributed_from"] == "mobil"
    assert ev5[0]["class_name"] == "truk/bus"
    assert 1 not in zc._car_lock_latched and 1 in zc._truck_lock_latched

    # Case B: Fast-track daytime truck (>= 1.5x eff_area AND >= 1.2x eff_h) promotes in 2 frames
    zc.update(_dets([[420, 60, 480, 120]], [2], tid=[2]), now=t + 10)
    ev_b = zc.update(_dets([[418, 80, 482, 145]], [2], tid=[2]), now=t + 11)
    assert len(ev_b) == 1 and ev_b[0]["class_name"] == "mobil"
    # Wide-only box (>= 1.5x area but short height < 1.2 * 50.4 = 60.5) must NOT fast-track
    d_short1 = _dets([[350, 200, 560, 255]], [7], conf=[0.9], tid=[2])  # 210x55 = 11550
    d_short1.data["perspective_truck"] = np.array([True], dtype=bool)
    assert not zc.update(d_short1, now=t + 12)
    # Full tall/large fast-track truck at mid y2=300: eff_area=12000 (1.5x=18000), eff_h=50.4 (1.2x=60.5)
    for k, box in enumerate(([370, 140, 520, 290], [365, 160, 525, 310])):  # 150x150=22500, 160x150=24000
        dt = _dets([box], [7], conf=[0.9], tid=[2])
        dt.data["perspective_truck"] = np.array([True], dtype=bool)
        ev_ft = zc.update(dt, now=t + 13 + k)
        if k == 0:
            assert not ev_ft
        else:
            assert len(ev_ft) == 1 and ev_ft[0]["reattributed_from"] == "mobil"
            assert ev_ft[0]["class_name"] == "truk/bus"
    assert zc.total == 2 and zc.total_by_class["truk/bus"] == 2
    assert zc.total_by_class.get("mobil", 0) == 0


def test_receding_truck_latched_permanent_through_far_zone():
    """Truck latched near bottom edge stays permanently in _truck_lock_latched
    and counted as truk/bus even as perspective shrinks < 0.85x effective
    truck area and raw class flickers to CAR_CLS in the upper/far zone."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [500, 0], [500, 500], [0, 500]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 5: "truk/bus", 7: "truk/bus"},
                     truck_min_area=12000.0)
    t = time.monotonic()

    # Frames 1..2: bottom entry (y2=460,410 > mid=250), fast-tracks into _truck_lock_latched
    for k, box in enumerate(([150, 280, 330, 460], [155, 235, 335, 410])):  # 180x180, 180x175
        d = _dets([box], [7], conf=[0.9], tid=[1])
        d.data["perspective_truck"] = np.array([True], dtype=bool)
        ev = zc.update(d, now=t + k)
        if k == 1:
            assert len(ev) == 1 and ev[0]["class_name"] == "truk/bus"
    assert 1 in zc._truck_lock_latched

    # Frames 3..8: moves upward (dy < 0) through upper/far zone while shrinking
    # below 0.85 * effective_truck_min_area and reading raw CAR_CLS (2)
    later = []
    for k in range(6):
        y2 = 230 - k * 25
        later += zc.update(
            _dets([[210, y2 - 65, 285, y2]], [2], conf=[0.95], tid=[1]),  # 75x65=4875 (< 0.85x eff)
            now=t + 2 + k,
        )
    assert not any(e.get("reattributed_from") for e in later), later
    assert 1 in zc._truck_lock_latched and 1 not in zc._car_lock_latched
    assert zc.total == 1
    assert zc.total_by_class["truk/bus"] == 1
    assert zc.total_by_class.get("mobil", 0) == 0


def test_approaching_car_suv_single_frame_truck_glitch_stays_mobil():
    """Approaching car/SUV with single-frame truk/bus glitches (or wide
    horizontal car crop) stays locked as mobil."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [600, 0], [600, 600], [0, 600]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 5: "truk/bus", 7: "truk/bus"},
                     truck_min_area=12000.0)
    t = time.monotonic()

    zc.update(_dets([[200, 160, 310, 270]], [2], tid=[1]), now=t)
    ev = zc.update(_dets([[200, 200, 320, 320]], [2], tid=[1]), now=t + 1)
    assert len(ev) == 1 and ev[0]["class_name"] == "mobil"

    # Single-frame truck glitch near bottom (even large & flagged), then back to car
    glitch = _dets([[180, 310, 340, 470]], [7], conf=[0.95], tid=[1])
    glitch.data["perspective_truck"] = np.array([True], dtype=bool)
    assert not zc.update(glitch, now=t + 2)
    assert not zc.update(_dets([[180, 340, 340, 500]], [2], conf=[0.9], tid=[1]), now=t + 3)
    # Another isolated single-frame glitch stays blocked
    glitch2 = _dets([[175, 370, 345, 540]], [7], conf=[0.95], tid=[1])
    glitch2.data["perspective_truck"] = np.array([True], dtype=bool)
    assert not zc.update(glitch2, now=t + 4)

    # Wide horizontal car crop (w/h = 280/100 = 2.8 > 2.0) across multiple frames stays mobil
    zc.update(_dets([[100, 200, 220, 280]], [2], tid=[2]), now=t + 10)
    zc.update(_dets([[110, 220, 230, 300]], [2], tid=[2]), now=t + 11)
    for k in range(4):
        zc.update(_dets([[100, 250 + k * 20, 380, 350 + k * 20]], [7], conf=[0.95], tid=[2]),
                  now=t + 12 + k)

    assert zc.total == 2
    assert zc.total_by_class["mobil"] == 2
    assert zc.total_by_class.get("truk/bus", 0) == 0


def test_live_zone_polygon_update_hook_and_hot_reload_preserve_counts(monkeypatch, tmp_path):
    """Web UI update_zone hook and ConfigHotReloader polygon edits update
    ZoneCounter, Renderer, and Detector Y-bounds in-place without resetting counts."""
    from fastapi.testclient import TestClient

    import main as main_mod
    from src.api import server as srv
    from src.core.detector import Detector
    from src.ui.renderer import Renderer

    settings_file = tmp_path / "settings.yaml"
    nodes_file = tmp_path / "nodes.yaml"
    settings_file.write_text(
        "detector:\n  conf: 0.2\n  car_min_area: 3000\n  truck_min_area: 12000\n"
        "counter:\n  min_displacement_px: 12\n",
        encoding="utf-8",
    )
    nodes_file.write_text(
        "nodes:\n"
        "- id: node_live_zone\n"
        "  nama: Test Node\n"
        "  lat: -7.42\n"
        "  lng: 109.22\n"
        "  zones:\n"
        "  - name: zone\n"
        "    polygon:\n"
        "    - [100, 200]\n"
        "    - [500, 200]\n"
        "    - [500, 500]\n"
        "    - [100, 500]\n",
        encoding="utf-8",
    )

    class _FakeModel:
        def half(self):
            pass

    class _FakeRTDETR:
        def __init__(self, path):
            self.model = _FakeModel()

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("ultralytics.RTDETR", _FakeRTDETR)

    zone = sv.PolygonZone(polygon=np.array([[100, 200], [500, 200], [500, 500], [100, 500]], dtype=np.float32))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 3: "motor"}, min_displacement_px=12)
    renderer = Renderer([zone], counter=zc)
    det = Detector(device="cpu", model_size="rtdetr-l", polygons=[zone])
    det_over = {"polygons": [zone]}

    # Count 1 vehicle and keep an active track outside zone before live polygon edit
    t = time.monotonic()
    zc.update(_dets([[280, 220, 320, 260]], [2], tid=[1]), now=t)
    ev = zc.update(_dets([[280, 260, 320, 300]], [2], tid=[1]), now=t + 1)
    assert len(ev) == 1 and zc.total == 1 and zc.total_by_class["mobil"] == 1
    zc.update(_dets([[280, 40, 320, 80]], [2], tid=[2]), now=t + 2)  # active track above zone

    reloader = main_mod.ConfigHotReloader(
        settings_path=settings_file,
        nodes_path=nodes_file,
        det=det,
        check_interval_sec=2.0,
    )
    reloader.register_node("node_live_zone", zc, renderer, det_over)
    srv.set_node_frame_size("node_live_zone", 1000, 800)
    srv.register_zone_update_hook(reloader.apply_node_zones, nodes_path=nodes_file)

    try:
        client = TestClient(srv.app)
        # Web UI saves normalized polygon covering y=[120, 680] on 1000x800 frame
        res = client.post(
            "/api/v1/control/node_live_zone/update_zone",
            json={"points": [[0.2, 0.15], [0.8, 0.15], [0.8, 0.85], [0.2, 0.85]]},
        )
        assert res.status_code == 200 and res.json()["applied"] is True

        # Counts and active tracks preserved; ZoneCounter, Renderer, and Detector updated immediately
        assert zc.total == 1 and zc.total_by_class["mobil"] == 1
        assert 2 in zc._start
        assert zc.zones[0].polygon.tolist() == [[200.0, 120.0], [800.0, 120.0],
                                                [800.0, 680.0], [200.0, 680.0]]
        assert renderer.zones[0].polygon.tolist() == zc.zones[0].polygon.tolist()
        assert renderer.zone_anns[0].zone is zc.zones[0]
        assert det.y_ref_far == 120.0 and det.y_ref_near == 680.0
        assert det_over["polygons"][0] is zc.zones[0]

        # Renderer draws onto frame without error with updated polygon
        frame = np.zeros((800, 1000, 3), dtype=np.uint8)
        renderer.draw(frame, None, dict(zc.total_by_class))
        assert frame.sum() > 0

        # Manual nodes.yaml edit triggers ConfigHotReloader to apply new polygon within 2s
        nodes_file.write_text(
            "nodes:\n"
            "- id: node_live_zone\n"
            "  nama: Test Node\n"
            "  lat: -7.42\n"
            "  lng: 109.22\n"
            "  zones:\n"
            "  - name: zone\n"
            "    polygon:\n"
            "    - [150, 240]\n"
            "    - [850, 240]\n"
            "    - [850, 720]\n"
            "    - [150, 720]\n",
            encoding="utf-8",
        )
        future = os.path.getmtime(nodes_file) + 5.0
        os.utime(nodes_file, (future, future))

        assert reloader.check_and_reload(now=100.0) is True
        assert zc.total == 1 and zc.total_by_class["mobil"] == 1
        assert zc.zones[0].polygon.tolist() == [[150.0, 240.0], [850.0, 240.0],
                                                [850.0, 720.0], [150.0, 720.0]]
        assert renderer.zone_anns[0].zone is zc.zones[0]
        assert det.y_ref_far == 240.0 and det.y_ref_near == 720.0
    finally:
        srv.register_zone_update_hook(None)
        srv.clear_commands("node_live_zone")
        with srv._lock:
            srv._node_frame_sizes.pop("node_live_zone", None)


def test_get_track_diagnostics_thread_safe_under_concurrent_updates():
    """Concurrent Renderer diagnostic reads must never hit deque mutated during iteration."""
    import threading

    zone = sv.PolygonZone(polygon=np.array([[0, 0], [500, 0], [500, 500], [0, 500]], dtype=np.float32))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 3: "motor", 7: "truk/bus"}, min_displacement_px=0)
    errors = []
    stop = threading.Event()

    def writer():
        t0 = time.monotonic()
        try:
            for step in range(400):
                cls = (2, 3, 7)[step % 3]
                zc.update(_dets([[100, 100 + (step % 40), 160, 160 + (step % 40)]], [cls], tid=[1]), now=t0 + step * 0.04)
                if step % 150 == 149:
                    zc.reset()
        except Exception as exc:
            errors.append(exc)
        finally:
            stop.set()

    def reader():
        try:
            while not stop.is_set():
                zc.get_track_diagnostics(1, [100, 100, 160, 160])
                zc.diagnostics_summary()
        except Exception as exc:
            errors.append(exc)

    t_w = threading.Thread(target=writer)
    t_r = threading.Thread(target=reader)
    t_r.start()
    t_w.start()
    t_w.join(timeout=5.0)
    t_r.join(timeout=5.0)
    assert not errors


def test_persist_zone_and_hot_reload_attribute_integrity(tmp_path):
    """persist_zone keeps custom zone attrs/extra zones; update_polygon/set_zones clear omitted rules."""
    import yaml
    import main as main_mod
    from src.utils.zone_store import persist_zone

    nodes_file = tmp_path / "custom_nodes.yaml"
    nodes_file.write_text(
        "nodes:\n"
        "- id: node_x\n"
        "  nama: Test\n"
        "  zones:\n"
        "  - name: south_gate\n"
        "    polygon: [[100, 100], [400, 100], [400, 400], [100, 400]]\n"
        "    allowed_directions: [\"down\"]\n"
        "    flow_max_angle_deg: 60\n"
        "    min_displacement_px: 15\n"
        "  - name: north_gate\n"
        "    polygon: [[500, 100], [700, 100], [700, 400], [500, 400]]\n",
        encoding="utf-8",
    )

    persist_zone("node_x", [[120, 120], [420, 120], [420, 420], [120, 420]], path=nodes_file)
    saved = yaml.safe_load(nodes_file.read_text(encoding="utf-8"))["nodes"][0]
    assert len(saved["zones"]) == 2
    assert saved["zones"][0]["name"] == "south_gate"
    assert saved["zones"][0]["polygon"] == [[120, 120], [420, 120], [420, 420], [120, 420]]
    assert saved["zones"][0]["allowed_directions"] == ["down"]
    assert saved["zones"][0]["flow_max_angle_deg"] == 60
    assert saved["zones"][0]["min_displacement_px"] == 15
    assert saved["zones"][1]["name"] == "north_gate"

    # Single-zone update_polygon clears omitted allowed_directions unless preserve_zone_attrs=True
    zc = ZoneCounter([{"name": "z0", "polygon": [[0, 0], [500, 0], [500, 500], [0, 500]],
                       "allowed_directions": ["down"], "flow_max_angle_deg": 45.0}],
                     class_labels={2: "mobil"}, min_displacement_px=12, dedup_cooldown_sec=0)
    t0 = time.monotonic()
    zc.update(_dets([[100, 300, 140, 340]], [2], tid=[1]), now=t0)
    assert zc.update(_dets([[100, 260, 140, 300]], [2], tid=[1]), now=t0 + 0.1) == []
    assert zc.suppressed_direction == 1

    zc.update_polygon([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]], 1000, 1000)
    assert not hasattr(zc.zones[0], "allowed_directions")
    zc.update(_dets([[100, 300, 140, 340]], [2], tid=[2]), now=t0 + 1.0)
    ev = zc.update(_dets([[100, 260, 140, 300]], [2], tid=[2]), now=t0 + 1.1)
    assert len(ev) == 1

    # set_zones and ConfigHotReloader also clear direction rules when deleted from nodes.yaml
    zc.set_zones([{"name": "z0", "polygon": [[0, 0], [500, 0], [500, 500], [0, 500]],
                   "allowed_directions": ["down"]}])
    reloader = main_mod.ConfigHotReloader(settings_path=None, nodes_path=nodes_file, check_interval_sec=0.0)
    reloader.register_node("node_x", zc)
    nodes_file.write_text(
        "nodes:\n"
        "- id: node_x\n"
        "  nama: Test\n"
        "  zones:\n"
        "  - name: south_gate\n"
        "    polygon: [[0, 0], [500, 0], [500, 500], [0, 500]]\n",
        encoding="utf-8",
    )
    future = os.path.getmtime(nodes_file) + 5.0
    os.utime(nodes_file, (future, future))
    assert reloader.check_and_reload(now=100.0) is True
    assert not hasattr(zc.zones[0], "allowed_directions")
    zc.update(_dets([[100, 300, 140, 340]], [2], tid=[3]), now=t0 + 2.0)
    ev2 = zc.update(_dets([[100, 260, 140, 300]], [2], tid=[3]), now=t0 + 2.1)
    assert len(ev2) == 1

    # Deferred _ai_worker update_zone forwards hot_reloader.nodes_path to persist_zone
    import threading
    from src.api import server as srv

    bus = main_mod.FrameBus()
    stop = threading.Event()
    prof = main_mod.StageProfiler("node_x")

    class _Det:
        def detect(self, f, **kw):
            return sv.Detections.empty()

        def configure_perspective_from_polygons(self, z):
            pass

    class _Str:
        def is_frozen(self):
            return False

        def effective_fps(self, default=25.0):
            return 25.0

    nodes_file.write_text(
        "nodes:\n- id: node_x\n  nama: Test\n  zones:\n"
        "  - name: south_gate\n    polygon: [[0, 0], [500, 0], [500, 500], [0, 500]]\n"
        "    allowed_directions: [\"down\"]\n",
        encoding="utf-8",
    )
    future2 = os.path.getmtime(nodes_file) + 10.0
    os.utime(nodes_file, (future2, future2))
    zc.update_polygon([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]], 1000, 1000,
                      allowed_directions=["down"])
    reloader.check_and_reload(now=200.0)
    srv.COMMAND_QUEUE["node_x"] = [{"action": "update_zone", "points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]}]
    th = threading.Thread(
        target=main_mod._ai_worker,
        args=("node_x", bus, _Str(), _Det(), zc, None, threading.Lock(),
              stop, prof, {"v": False}, time.monotonic()),
        kwargs={"hot_reloader": reloader},
        daemon=True,
    )
    th.start()
    try:
        bus.publish_raw(np.zeros((1000, 1000, 3), dtype=np.uint8), time.monotonic(), time.time())
        deadline = time.monotonic() + 3.0
        while srv.COMMAND_QUEUE.get("node_x") and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.05)
    finally:
        stop.set()
        th.join(timeout=3.0)
        srv.clear_commands("node_x")
    saved_after = yaml.safe_load(nodes_file.read_text(encoding="utf-8"))["nodes"][0]["zones"][0]
    assert saved_after["polygon"] == [[100, 100], [900, 100], [900, 900], [100, 900]]
    assert saved_after["allowed_directions"] == ["down"]
    assert zc.zones[0].allowed_directions == ["down"]


def test_mid_zone_car_id_split_velocity_and_bbox_dedup_no_double_count():
    """Mid-zone car splits over 8-frame gap (velocity-projected + box-scaled ghost handover)
    and coexisting 1-frame split handoff (vanished-track bbox IoU dedup) count only once."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [800, 0], [800, 600], [0, 600]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 3: "motor"},
                     dedup_cooldown_sec=0.0)
    assert zc.ghost_ttl_frames == 12
    t = time.monotonic()

    # Car tid=1 (120x90 -> box-scaled radius max(35, 0.65*120)=78px), moves +10px/frame right
    zc.update(_dets([[80, 200, 200, 290]], [2], tid=[1]), now=t)
    ev1 = zc.update(_dets([[95, 200, 215, 290]], [2], tid=[1]), now=t + 0.04)  # disp=15px -> counts
    assert len(ev1) == 1 and zc.total == 1
    zc.update(_dets([[105, 200, 225, 290]], [2], tid=[1]), now=t + 0.08)  # vel=(10,0), center=(165,245)

    # 8 empty frames (occluded by truck); car traveled ~80px right to center (245, 245)
    for k in range(8):
        zc.update(sv.Detections.empty(), now=t + 0.12 + k * 0.04)

    # tid=2 spawns near velocity-predicted (245, 245) + 55px shift on box edge -> ghost handover catches it
    zc.update(_dets([[220, 205, 340, 295]], [2], tid=[2]), now=t + 0.44)
    ev2 = zc.update(_dets([[230, 205, 350, 295]], [2], tid=[2]), now=t + 0.48)
    assert not ev2 and zc.total == 1 and zc.ghost_handovers == 1

    # Next: coexisting 1-frame split where tid=3 overlaps tid=2 BEFORE tid=2 vanishes
    # (ghost handover won't match because tid=3 coexisted with tid=2; vanished-track IoU catches it)
    zc.update(_dets([[240, 205, 360, 295], [248, 208, 368, 298]], [2, 2], tid=[2, 3]), now=t + 0.52)
    # tid=2 gone on next frame; tid=3 moves +15px from its birth -> vanished-track bbox dedup blocks
    ev3 = zc.update(_dets([[263, 208, 383, 298]], [2], tid=[3]), now=t + 0.56)
    assert not ev3 and zc.total == 1 and zc.suppressed == 1
    assert 3 in zc._counted[0]
    assert zc.total_by_class["mobil"] == 1

    zc.reset()
    assert not any(zc._recent_counted_tracks)


def test_dense_side_by_side_cars_never_suppress_each_other():
    """Two distinct active cars side-by-side in the same frames both count."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [800, 0], [800, 600], [0, 600]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil"}, min_displacement_px=12.0)
    t = time.monotonic()

    # Frame 1: both cars enter zone side-by-side (~20px gap)
    zc.update(_dets([[100, 100, 220, 200], [230, 100, 350, 200]], [2, 2], tid=[1, 2]), now=t)
    # Frame 2: car 1 moves 20px, car 2 moves 18px; both active -> both must count!
    ev = zc.update(_dets([[100, 120, 220, 220], [230, 118, 350, 218]], [2, 2], tid=[1, 2]), now=t + 0.04)
    assert len(ev) == 2
    assert zc.total == 2 and zc.total_by_class["mobil"] == 2
    assert zc.suppressed == 0


def test_transient_one_frame_motor_split_box_blocked_by_min_track_frames():
    """Default min_track_frames=2 rejects transient 1-frame split boxes without delaying normal motors."""
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [500, 0], [500, 500], [0, 500]]))
    zc = ZoneCounter([zone], class_labels={2: "mobil", 3: "motor"},
                     min_displacement_px=0.0, dedup_cooldown_sec=0.0)
    assert zc.min_track_frames == 2
    t = time.monotonic()

    # Frame 1: real motor tid=1 + transient 1-frame split box tid=99
    assert zc.update(_dets([[100, 100, 125, 135], [130, 102, 152, 134]], [3, 3], tid=[1, 99]), now=t) == []
    # Frame 2: tid=99 vanished after 1 frame (never counted); tid=1 hits frame 2 -> counts immediately!
    ev = zc.update(_dets([[100, 120, 125, 155]], [3], tid=[1]), now=t + 0.04)
    assert len(ev) == 1 and ev[0]["track_id"] == 1
    assert zc.total == 1 and zc.total_by_class["motor"] == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
