"""Persistence + WIB midnight rollover checks for src/utils/stats_store.

Run: python -m pytest tests/test_stats_store.py
"""
import json
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import supervision as sv

from src.core.zone_counter import ZoneCounter
from src.utils.stats_store import (WIB, StatsStore, fold_hourly_counts,
                                   seed_counter, today_wib)


def _yesterday():
    return (datetime.now(WIB) - timedelta(days=1)).strftime("%Y-%m-%d")


def _snapshot_doc(date, total, counts, hourly=None):
    return {"date": date, "updated_at": "2000-01-01T00:00:00+07:00",
            "nodes": {"node_01": {"total": total, "counts": counts,
                                  "hourly_counts": hourly or {}}}}


class _FakeCounter:
    """The two attributes StatsStore reads off a ZoneCounter."""

    def __init__(self, counts=None, total=0):
        self.total = total
        self.total_by_class = defaultdict(int, counts or {})


def _zone_counter():
    zone = sv.PolygonZone(
        polygon=np.array([[0, 0], [200, 0], [200, 200], [0, 200]]))
    return ZoneCounter([zone], class_labels={2: "mobil", 3: "motor",
                                             5: "truk/bus", 7: "truk/bus"},
                       min_displacement_px=0, dedup_radius_px=0)


def _dets(xyxy, cls, tid):
    return sv.Detections(
        xyxy=np.asarray(xyxy, dtype=np.float32),
        confidence=np.full(len(xyxy), 0.9, dtype=np.float32),
        class_id=np.asarray(cls, dtype=np.int32),
        tracker_id=np.asarray(tid, dtype=np.int64))


def test_save_and_load_same_day(tmp_path):
    path = tmp_path / "daily_stats.json"
    store = StatsStore(path, save_interval_sec=3600)
    counts = {"motor": 4, "mobil": 2, "truk/bus": 1}
    hourly = {"08": {"motor": 4, "mobil": 2, "truk": 1}}
    store.record("node_01", _FakeCounter(counts, 7), hourly)
    assert store.maybe_save(force=True)

    # unchanged state is not marked dirty; a change inside the interval waits
    assert store.record("node_01", _FakeCounter(counts, 7), hourly) is False
    store.record("node_01", _FakeCounter(counts, 8), hourly)
    assert store.maybe_save() is False
    store.save()

    restored = StatsStore(path)
    state = restored.load()
    assert state["date"] == today_wib()
    assert state["updated_at"].endswith("+07:00")
    node = restored.node("node_01")
    assert node["total"] == 8
    assert node["counts"] == {"motor": 4, "mobil": 2, "truk": 1}
    assert node["hourly_counts"] == hourly
    assert not (tmp_path / "daily_stats.json.tmp").exists()


def test_startup_rollover_archives_yesterday(tmp_path):
    path = tmp_path / "daily_stats.json"
    old = _yesterday()
    path.write_text(json.dumps(_snapshot_doc(
        old, 9, {"motor": 9, "mobil": 0, "truk": 0},
        {"23": {"motor": 9, "mobil": 0, "truk": 0}})), encoding="utf-8")

    store = StatsStore(path)
    store.load()
    assert store.node("node_01")["total"] == 0
    assert store.node("node_01")["hourly_counts"] == {}
    assert store.state["date"] == today_wib()

    archive = tmp_path / "archive" / f"daily_stats_{old}.json"
    assert archive.exists()
    assert json.loads(archive.read_text(encoding="utf-8"))["nodes"][
        "node_01"]["total"] == 9


def test_live_midnight_transition(tmp_path):
    path = tmp_path / "daily_stats.json"
    store = StatsStore(path)
    counter = _zone_counter()
    counter.update(_dets([[90, 90, 110, 110]], [2], [1]))
    hourly = {"23": {"motor": 0, "mobil": 1, "truk": 0}}
    store.record("node_01", counter, hourly)
    store.save()

    # the clock crosses midnight: the stored date is now yesterday
    old = _yesterday()
    store.state["date"] = old
    roll = store.rollover()
    assert roll is not None and roll["from"] == old and roll["to"] == today_wib()
    archive = tmp_path / "archive" / f"daily_stats_{old}.json"
    assert json.loads(archive.read_text(encoding="utf-8"))["nodes"][
        "node_01"]["total"] == 1
    assert store.rollover() is None            # same day again: no-op

    # worker sequence on rollover: ZoneCounter.reset + hourly clear + persist
    counter.reset()
    hourly.clear()
    assert counter.total == 0 and not counter.total_by_class
    store.record("node_01", counter, hourly)
    store.save()

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["date"] == today_wib()
    assert on_disk["nodes"]["node_01"]["total"] == 0
    assert on_disk["nodes"]["node_01"]["hourly_counts"] == {}


def test_corrupted_file_recovers(tmp_path):
    path = tmp_path / "daily_stats.json"
    path.write_text("{ this is not json", encoding="utf-8")
    store = StatsStore(path)
    state = store.load()                       # must not raise
    assert state["nodes"] == {}
    assert state["date"] == today_wib()

    store.record("node_01", _FakeCounter({"motor": 2}, 2), {})
    store.save()
    assert json.loads(path.read_text(encoding="utf-8"))["nodes"][
        "node_01"]["total"] == 2


def test_restored_counts_continue_and_reattribute(tmp_path):
    path = tmp_path / "daily_stats.json"
    store = StatsStore(path)
    store.record("node_01",
                 _FakeCounter({"motor": 5, "mobil": 3, "truk/bus": 1}, 9),
                 {"07": {"motor": 5, "mobil": 3, "truk": 1}})
    store.save()

    restored = StatsStore(path)
    restored.load()
    counter = _zone_counter()
    seed_counter(counter, restored.node("node_01"))
    assert counter.total == 9
    assert (counter.total_by_class["motor"], counter.total_by_class["mobil"],
            counter.total_by_class["truk/bus"]) == (5, 3, 1)

    # new detections keep incrementing from the restored baseline
    counters_before = dict(restored.node("node_01")["counts"])
    events = counter.update(_dets([[90, 90, 110, 110]], [2], [42]))
    assert len(events) == 1 and counter.total == 10
    assert counter.total_by_class["mobil"] == counters_before["mobil"] + 1
    hourly = dict(restored.node("node_01")["hourly_counts"])
    fold_hourly_counts(hourly, events, hour="08")

    # a mid-zone class flip re-attributes: bucket moves, total never changes
    re = counter._reattribute(0, 42, 5, 0.9, 100.0, 100.0, 0.0)
    assert re["reattributed_from"] == "mobil" and counter.total == 10
    fold_hourly_counts(hourly, [re], hour="08")
    assert hourly["08"] == {"motor": 0, "mobil": 0, "truk": 1}
    assert hourly["07"] == {"motor": 5, "mobil": 3, "truk": 1}

    store.record("node_01", counter, hourly)
    store.save()
    reread = StatsStore(path)
    reread.load()
    node = reread.node("node_01")
    assert node["total"] == 10
    assert node["counts"] == {"motor": 5, "mobil": 3, "truk": 2}
    assert node["hourly_counts"]["08"] == hourly["08"]


def test_api_state_seeded_before_first_frame(tmp_path):
    """Dashboard shows restored counts while the node has no live frames."""
    from src.api.server import nodes_state, seed_node_data

    path = tmp_path / "daily_stats.json"
    store = StatsStore(path)
    store.record("node_seed_test",
                 _FakeCounter({"motor": 4, "mobil": 2, "truk/bus": 1}, 7),
                 {"08": {"motor": 4, "mobil": 2, "truk": 1}})
    store.save()
    restored = StatsStore(path)
    restored.load()
    node = restored.node("node_seed_test")
    seed_node_data("node_seed_test", {
        "motor": node["counts"]["motor"], "mobil": node["counts"]["mobil"],
        "truk": node["counts"]["truk"], "hourly_counts": node["hourly_counts"]})

    st = nodes_state["node_seed_test"]
    assert (st["motor"], st["mobil"], st["truk"]) == (4, 2, 1)
    assert st["hourly_counts"]["08"] == {"motor": 4, "mobil": 2, "truk": 1,
                                         "total": 7}
    assert st["last_seen"] == 0.0   # offline until the worker reports
