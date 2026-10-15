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
                     dedup_cooldown_sec=4.0, min_displacement_px=0)
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
                     dedup_cooldown_sec=4.0, min_displacement_px=0)
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
    zc = ZoneCounter([zone], class_labels={3: "motor"})  # default gate = 20px
    t = time.monotonic()
    for i in range(10):
        assert not zc.update(_dets([[90, 90, 110, 110]], [3], tid=[1]), now=t + i)
    assert zc.total == 0


def test_moving_object_counted():
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [400, 0], [400, 400], [0, 400]]))
    zc = ZoneCounter([zone], class_labels={3: "motor"})  # default gate = 20px
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
                     min_displacement_px=50, still_count_frames=5, still_min_conf=0.35)
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
                     min_displacement_px=50, still_count_frames=5, still_min_conf=0.35)
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
                     min_displacement_px=0)
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
    zc = ZoneCounter([zone], class_labels={3: "motor"}, min_displacement_px=0)
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
                     min_displacement_px=0)
    t = time.monotonic()
    ev = zc.update(_dets([[90, 90, 130, 130]], [3], tid=[1],
                         demoted_car=[True]), now=t)
    assert len(ev) == 1 and ev[0]["class_name"] == "motor", ev


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
