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


def test_min_persist_uptime_gate_preserves_baseline_until_threshold(tmp_path, monkeypatch):
    """Short test sessions (< 1800s) update in-memory get_all() but leave
    daily_stats.json untouched at _baseline_state until uptime >= 1800s."""
    path = tmp_path / "daily_stats.json"
    baseline_store = StatsStore(path)
    baseline_hourly = {"08": {"motor": 20, "mobil": 8, "truk": 2}}
    baseline_store.record("node_01",
                          _FakeCounter({"motor": 20, "mobil": 8, "truk/bus": 2}, 30),
                          baseline_hourly)
    baseline_store.save(force=True)

    clock = [1000.0]
    monkeypatch.setattr("src.utils.stats_store.time.monotonic", lambda: clock[0])

    store = StatsStore.from_config({
        "stats_file": str(path),
        "save_interval_sec": 5.0,
        "min_persist_uptime_seconds": 1800.0,
    })
    store.load()
    assert store._baseline_state["nodes"]["node_01"]["total"] == 30

    # At t = 120s (2 minutes), 5 test vehicles are counted in hour 13
    clock[0] = 1120.0
    session_hourly = {
        "08": {"motor": 20, "mobil": 8, "truk": 2},
        "13": {"motor": 3, "mobil": 2, "truk": 0},
    }
    store.record("node_01",
                 _FakeCounter({"motor": 23, "mobil": 10, "truk/bus": 2}, 35),
                 session_hourly)

    # In-memory state (get_all) reflects real-time counts immediately
    mem = store.get_all()["nodes"]["node_01"]
    assert mem["total"] == 35
    assert mem["hourly_counts"]["13"] == {"motor": 3, "mobil": 2, "truk": 0}

    # Periodic maybe_save() and early shutdown save() before 1800s do NOT overwrite disk
    assert store.maybe_save() is False
    assert store.save() is False
    on_disk_early = json.loads(path.read_text(encoding="utf-8"))["nodes"]["node_01"]
    assert on_disk_early["total"] == 30
    assert "13" not in on_disk_early["hourly_counts"]

    # Once session uptime reaches 1800s, maybe_save() persists increments to disk
    clock[0] = 2800.0
    assert store.maybe_save() is True
    on_disk_late = json.loads(path.read_text(encoding="utf-8"))["nodes"]["node_01"]
    assert on_disk_late["total"] == 35
    assert on_disk_late["hourly_counts"]["13"] == {"motor": 3, "mobil": 2, "truk": 0}


def test_prune_hourly_below_and_clear(tmp_path):
    """prune_hourly_below(20) removes tiny hourly buckets (< 20 vehicles) and
    recomputes node daily totals; clear() resets persisted state."""
    path = tmp_path / "daily_stats.json"
    store = StatsStore(path)
    hourly = {
        "08": {"motor": 20, "mobil": 10, "truk": 5},   # 35 >= 20 -> keep
        "13": {"motor": 3, "mobil": 2, "truk": 0},     # 5 < 20 -> prune
    }
    store.record("node_01",
                 _FakeCounter({"motor": 23, "mobil": 12, "truk/bus": 5}, 40),
                 hourly)
    store.save(force=True)

    pruned = store.prune_hourly_below(20)
    assert pruned == 1
    store.save(force=True)

    reread = StatsStore(path)
    reread.load()
    node = reread.node("node_01")
    assert "13" not in node["hourly_counts"]
    assert node["hourly_counts"] == {"08": {"motor": 20, "mobil": 10, "truk": 5}}
    assert node["counts"] == {"motor": 20, "mobil": 10, "truk": 5}
    assert node["total"] == 35

    reread.clear(save_to_disk=True)
    assert reread.get_all()["nodes"] == {}
    assert json.loads(path.read_text(encoding="utf-8"))["nodes"] == {}


def test_midnight_rollover_with_uptime_gate_persists_reset_and_gates_new_day(tmp_path, monkeypatch):
    """Session started 10 min before midnight (min_persist_uptime_seconds=1800):
    midnight rollover archives yesterday, force-saves reset day to disk, updates
    _baseline_state, and keeps post-midnight counts in memory until 30 min uptime."""
    day1 = "2026-11-15"
    day2 = "2026-11-16"
    day3 = "2026-11-17"
    current_day = [day1]
    clock = [1000.0]

    monkeypatch.setattr("src.utils.stats_store.today_wib", lambda: current_day[0])
    monkeypatch.setattr("src.utils.stats_store.time.monotonic", lambda: clock[0])

    path = tmp_path / "daily_stats.json"
    path.write_text(
        json.dumps(_snapshot_doc(
            day1, 40, {"motor": 30, "mobil": 8, "truk": 2},
            {"22": {"motor": 30, "mobil": 8, "truk": 2}},
        )),
        encoding="utf-8",
    )

    # Start session 10 minutes before midnight (23:50 WIB)
    store = StatsStore.from_config({
        "stats_file": str(path),
        "save_interval_sec": 5.0,
        "min_persist_uptime_seconds": 1800.0,
    })
    store.load()

    # Pre-midnight activity at +5 min (uptime 300s < 1800s): visible in memory, gated on disk
    clock[0] = 1300.0
    store.record(
        "node_01",
        _FakeCounter({"motor": 33, "mobil": 9, "truk/bus": 2}, 44),
        {"22": {"motor": 30, "mobil": 8, "truk": 2},
         "23": {"motor": 3, "mobil": 1, "truk": 0}},
    )
    assert store.maybe_save() is False

    # Cross 00:00 WIB at +10 min (uptime 600s < 1800s)
    clock[0] = 1600.0
    current_day[0] = day2
    roll = store.rollover()
    assert roll is not None and roll["from"] == day1 and roll["to"] == day2

    # Yesterday's snapshot archived (with session state at rollover) and disk reset immediately
    archive = tmp_path / "archive" / f"daily_stats_{day1}.json"
    assert archive.exists()
    assert json.loads(archive.read_text(encoding="utf-8"))["nodes"]["node_01"]["total"] == 44

    on_disk_midnight = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk_midnight["date"] == day2
    assert on_disk_midnight["nodes"] == {}
    assert store._baseline_state["date"] == day2
    assert store._baseline_state["nodes"] == {}

    # Post-midnight counts at +15 min (uptime 900s < 1800s): held in memory, NOT on disk
    clock[0] = 1900.0
    store.record(
        "node_01",
        _FakeCounter({"motor": 4, "mobil": 2, "truk/bus": 0}, 6),
        {"00": {"motor": 4, "mobil": 2, "truk": 0}},
    )
    assert store.get_all()["nodes"]["node_01"]["total"] == 6
    assert store.maybe_save() is False
    assert store.save() is False
    assert json.loads(path.read_text(encoding="utf-8"))["nodes"] == {}

    # Once 30 min uptime is reached (+30 min = 2800s), post-midnight counts persist
    clock[0] = 2800.0
    assert store.maybe_save() is True
    on_disk_after = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk_after["date"] == day2
    assert on_disk_after["nodes"]["node_01"]["total"] == 6

    # Also verify load() rollover when file on disk is older than today under uptime gate
    current_day[0] = day3
    clock[0] = 3000.0
    store_restart = StatsStore(path, save_interval_sec=5.0, min_persist_uptime_seconds=1800.0)
    store_restart.load()
    assert (tmp_path / "archive" / f"daily_stats_{day2}.json").exists()
    on_disk_day3 = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk_day3["date"] == day3
    assert on_disk_day3["nodes"] == {}
    assert store_restart._baseline_state["date"] == day3
    assert store_restart._baseline_state["nodes"] == {}
