"""Smoke tests for the refactored core: detector NMS + ZoneCounter trigger/dedup.
Run:  python tests/test_counter.py  (repo root on PYTHONPATH)"""
import csv
import os
import tempfile

import numpy as np
import supervision as sv

from src.core.detector import cross_class_nms
from src.core.zone_counter import ZoneCounter


def make_zone(poly):
    return sv.PolygonZone(polygon=np.array(poly, dtype=np.float32))


def make_dets(xyxy, cls=3, tid=1):
    xyxy = np.array(xyxy, dtype=float).reshape(-1, 4)
    n = len(xyxy)
    tids = np.atleast_1d(np.array(tid, dtype=int))
    if len(tids) == 1:
        tids = np.repeat(tids, n)
    return sv.Detections(
        xyxy=xyxy,
        class_id=np.full(n, cls, dtype=int),
        tracker_id=tids,
        confidence=np.ones(n),
    )


Z = [(100, 200), (600, 200), (600, 500), (100, 500)]

# 1: centroid inside -> counted immediately, no movement needed
c = ZoneCounter([make_zone(Z)], {3: "motor"}, min_displacement_px=0)
c.update(make_dets([[300, 250, 400, 350]], tid=1), now=0.0)
assert c.total == 1, f"1 FAIL={c.total}"

# 2: same id stays inside for minutes -> still 1 (permanent latch)
for t in range(1, 120):
    c.update(make_dets([[300, 250, 400, 350]], tid=1), now=float(t) * 0.1)
assert c.total == 1, f"2 FAIL={c.total}"

# 3: id exits and re-enters -> not recounted (latch is permanent per zone)
c.update(make_dets([[20, 250, 80, 350]], tid=1), now=20.0)
c.update(make_dets([[300, 250, 400, 350]], tid=1), now=20.1)
assert c.total == 1, f"3 FAIL={c.total}"

# 4: new id inside -> counted (fresh vehicle), instant class from current frame
c = ZoneCounter([make_zone(Z)], {2: "mobil", 3: "motor"}, min_displacement_px=0)
c.update(make_dets([[300, 250, 400, 350]], cls=2, tid=5), now=0.0)
assert c.total == 1 and c.total_by_class["mobil"] == 1, f"4 FAIL={dict(c.total_by_class)}"

# 5: jitter re-ID inside near a fresh count -> suppressed + aliased
c = ZoneCounter([make_zone(Z)], {3: "motor"}, dedup_cooldown_sec=4.0, dedup_radius_px=40,
                min_displacement_px=0)
c.update(make_dets([[300, 250, 400, 350]], tid=1), now=0.0)
c.update(make_dets([[305, 252, 405, 352]], tid=2), now=0.2)   # ghost re-ID same spot
assert c.total == 1 and c.suppressed == 1, f"5 FAIL total={c.total} sup={c.suppressed}"

# 6: after cooldown, same spot new id -> genuine count
c.update(make_dets([[300, 250, 400, 350]], tid=3), now=5.0)
assert c.total == 2, f"6 FAIL={c.total}"

# 7: outside zone never counts
c = ZoneCounter([make_zone(Z)], {3: "motor"}, min_displacement_px=0)
c.update(make_dets([[20, 250, 80, 350]], tid=1), now=0.0)
assert c.total == 0, f"7 FAIL={c.total}"

# 8: two zones, one vehicle inside both -> one count per zone
za = make_zone([(100, 200), (350, 200), (350, 500), (100, 500)])
zb = make_zone([(340, 200), (600, 200), (600, 500), (340, 500)])
c = ZoneCounter([za, zb], {3: "motor"}, dedup_cooldown_sec=0.0,
                min_displacement_px=0)  # no dedup/gate for this case
c.update(make_dets([[300, 300, 400, 400]], tid=1), now=0.0)      # centroid 350,350 in both
assert c.total == 2, f"8 FAIL={c.total}"

# 9: freeze (jitter grace) -> no counting, unfreeze -> counts again
c = ZoneCounter([make_zone(Z)], {3: "motor"}, min_displacement_px=0)
c.update(make_dets([[300, 250, 400, 350]], tid=1), now=0.0, freeze=True)
assert c.total == 0, f"9 FAIL={c.total}"
c.update(make_dets([[300, 250, 400, 350]], tid=1), now=2.0)
assert c.total == 1, f"9b FAIL={c.total}"

# 10: empty detections -> no crash
c.update(sv.Detections.empty())
assert c.total == 1

# 11: reset clears latches
c.reset()
assert c.total == 0
c.update(make_dets([[300, 250, 400, 350]], tid=1), now=3.0)
assert c.total == 1, f"11 FAIL={c.total}"

# 12: count events returned for the audit logger (one per accepted count)
c = ZoneCounter([make_zone(Z)], {3: "motor"}, min_displacement_px=0)
ev = c.update(make_dets([[300, 250, 400, 350]], tid=7), now=0.0)
assert len(ev) == 1 and ev[0]["track_id"] == 7 and ev[0]["class_name"] == "motor", f"12 FAIL={ev}"
assert 7 in c.counted_tracks_by_class["motor"], "12 class-mapped memory FAIL"

# 13: AuditLogger writes header + one row per event
from src.utils.logger import AuditLogger
tmp = os.path.join(tempfile.mkdtemp(), "count_audit.csv")
al = AuditLogger(tmp)
al.write(ev[0])
al.close()
rows = list(csv.reader(open(tmp)))
assert rows[0] == ["timestamp", "track_id", "class_name", "confidence",
                   "centroid_x", "centroid_y"], f"13 header FAIL={rows[0]}"
assert len(rows) == 2 and rows[1][1] == "7" and rows[1][2] == "motor", f"13 FAIL={rows}"

# 14: cross-class NMS -> car beats overlapping motorcycle
d = sv.Detections(
    xyxy=np.array([[0, 0, 100, 100], [10, 10, 110, 110]], dtype=np.float32),
    class_id=np.array([3, 2], dtype=int),
    confidence=np.array([0.9, 0.9], dtype=np.float32),
)
kept = cross_class_nms(d, duplicate_iou=0.5)
assert list(kept.class_id) == [2], f"14 FAIL={kept.class_id}"

print("ALL 14 TESTS PASS")
