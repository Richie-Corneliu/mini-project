"""Core-logic checks for cross_class_nms + ZoneCounter (no GPU needed).
Run:  python -m tests.test_zone_counter  (repo root on PYTHONPATH)"""
import time

import numpy as np
import supervision as sv

from src.core.detector import cross_class_nms
from src.core.zone_counter import ZoneCounter


def _dets(xyxy, cls, conf=None, tid=None):
    n = len(xyxy)
    return sv.Detections(
        xyxy=np.asarray(xyxy, dtype=np.float32),
        confidence=np.asarray(conf if conf is not None else [0.9] * n, dtype=np.float32),
        class_id=np.asarray(cls, dtype=np.int32),
        tracker_id=np.asarray(tid, dtype=np.int64) if tid is not None else None,
    )


def test_nms_car_beats_motorcycle():
    d = _dets([[0, 0, 100, 100], [10, 10, 110, 110]], [3, 2])  # motor first, car bigger-priority
    kept = cross_class_nms(d, duplicate_iou=0.5)
    assert list(kept.class_id) == [2], kept.class_id


def test_nms_tie_keeps_bigger():
    d = _dets([[0, 0, 100, 100], [5, 5, 60, 60]], [2, 2])
    kept = cross_class_nms(d, duplicate_iou=0.3)
    assert len(kept) == 1
    assert kept.xyxy[0][2] == 100


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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
