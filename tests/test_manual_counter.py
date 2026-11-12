"""Unit tests for the live AI-synced benchmark counter (scripts/manual_counter.py).

Everything here runs headless: no cv2 window is opened, the API poller thread
is never started, and no file leaves tmp_path.
"""
import json
import urllib.error

import numpy as np
import pytest

from scripts import manual_counter as mc
from scripts.manual_counter import (CLASS_NAMES, H, W, BenchmarkSession,
                                    handle_key, parse_args, render)


def payload(nodes):
    return {"nodes": nodes}


def node(motor=0, mobil=0, truk=0, nama="Simpang 01"):
    return {"nama": nama, "motor": motor, "mobil": mobil, "truk": truk}


def fake_clock(start=1000.0):
    box = [start]
    return box, (lambda: box[0])


def make_session(**kw):
    box, clock = fake_clock(kw.pop("start", 1000.0))
    s = BenchmarkSession(clock=clock, **kw)
    return s, box


# ---- 1. counts -------------------------------------------------------

def test_increment_decrement_clamped_and_reset():
    s, _ = make_session()
    for name in CLASS_NAMES:
        assert s.counts[name] == 0
        s.increment(name)
        s.increment(name)
        assert s.counts[name] == 2
        s.decrement(name)
        assert s.counts[name] == 1
        s.decrement(name)
        s.decrement(name)          # never goes negative
        assert s.counts[name] == 0
    s.increment("motor")
    s.increment("bis/truk")
    assert s.manual_total == 2

    s.start_session()
    s.reset()
    assert s.counts == {name: 0 for name in CLASS_NAMES}
    assert s.state == "IDLE"
    assert s.started_at is None and s.ai_baseline is None
    assert s.ai_delta() == mc.zeros()


# ---- 2. session lifecycle + AI delta ---------------------------------

def test_session_lifecycle_baseline_delta_and_freeze():
    s, clock = make_session(node_id="simpang_01")

    assert s.apply_payload(payload({"simpang_01": node(10, 5, 2)}))
    assert s.api_connected
    assert s.ai_latest == {"motor": 10, "mobil": 5, "truk_bus": 2}
    assert s.node_name == "Simpang 01"
    assert s.state == "IDLE"
    assert s.ai_delta() == mc.zeros()          # no baseline yet

    s.start_session()
    assert s.state == "RUNNING"
    assert s.ai_baseline == {"motor": 10, "mobil": 5, "truk_bus": 2}
    clock[0] += 30
    assert s.elapsed_seconds() == pytest.approx(30)

    s.apply_payload(payload({"simpang_01": node(18, 9, 5)}))
    assert s.ai_delta() == {"motor": 8, "mobil": 4, "truk_bus": 3}

    s.apply_payload(payload({"simpang_01": node(3, 1, 1)}))   # counters reset
    assert s.ai_delta() == {"motor": 0, "mobil": 0, "truk_bus": 0}  # clamped

    clock[0] += 30
    s.stop_session()
    assert s.state == "STOPPED"
    assert s.elapsed_seconds() == pytest.approx(60)
    frozen = s.elapsed_seconds()

    s.apply_payload(payload({"simpang_01": node(18, 9, 5)}))
    assert s.ai_delta() == {"motor": 0, "mobil": 0, "truk_bus": 0}  # frozen
    assert s.elapsed_seconds() == pytest.approx(frozen)

    s.toggle_session()                          # STOPPED -> RUNNING again
    assert s.state == "RUNNING"
    # new baseline is the current live counter, so the delta restarts at zero
    assert s.ai_baseline == {"motor": 18, "mobil": 9, "truk_bus": 5}
    assert s.ai_delta() == mc.zeros()


def test_toggle_and_cycle_node_and_auto_select():
    s, _ = make_session()                       # node_id=None -> first sorted
    two = payload({"simpang_02": node(1, 1, 1, "Dua"),
                   "simpang_01": node(2, 2, 2, "Satu")})
    s.apply_payload(two)
    assert s.node_ids == ["simpang_01", "simpang_02"]
    assert s.node_id == "simpang_01"

    assert s.cycle_node() == "simpang_02"
    assert s.node_name == "Dua"
    assert s.ai_latest == {"motor": 1, "mobil": 1, "truk_bus": 1}
    assert s.cycle_node() == "simpang_01"       # wraps around

    s.toggle_session()                          # RUNNING
    assert s.cycle_node() == "simpang_01"       # locked while running
    s.toggle_session()                          # STOPPED
    assert s.cycle_node() == "simpang_01"

    single = BenchmarkSession()
    single.apply_payload(payload({"only_one": node()}))
    assert single.cycle_node() == "only_one"    # nothing to cycle


# ---- 3. live accuracy via scripts/evaluate_accuracy.py ---------------

def test_live_report_matches_evaluate_accuracy():
    s, _ = make_session(node_id="simpang_01")
    s.apply_payload(payload({"simpang_01": node(0, 0, 0)}))
    s.start_session()
    s.apply_payload(payload({"simpang_01": node(10, 8, 4)}))

    s.counts.update({"motor": 10, "mobil": 8, "bis/truk": 5})
    rep = s.live_report()

    assert rep["scenario"] == "Live RTSP Validation"
    assert rep["gt_total"] == 23 and rep["pred_total"] == 22
    by_key = {r["key"]: r for r in rep["rows"]}
    assert by_key["motor"]["accuracy"] == pytest.approx(100.0)
    assert by_key["mobil"]["accuracy"] == pytest.approx(100.0)
    assert by_key["truk_bus"]["accuracy"] == pytest.approx(80.0)   # 1 miss / 5
    assert by_key["total"]["gt"] == 23 and by_key["total"]["pred"] == 22
    assert rep["weighted_accuracy"] == pytest.approx((1 - 1 / 23) * 100)
    assert rep["macro_accuracy"] == pytest.approx((100 + 100 + 80) / 3)
    assert rep["passed"] is True

    s.counts.update({"motor": 10, "mobil": 8, "bis/truk": 10})
    rep2 = s.live_report()                      # gt 28 vs pred 22 -> fail
    assert rep2["gt_total"] == 28 and rep2["pred_total"] == 22
    assert rep2["weighted_accuracy"] == pytest.approx((1 - 6 / 28) * 100)
    assert rep2["passed"] is False


def test_live_report_standalone_zero_counts():
    s, _ = make_session()                       # never connected
    rep = s.live_report()
    assert s.api_connected is False
    assert rep["gt_total"] == 0 and rep["pred_total"] == 0
    assert rep["weighted_accuracy"] == pytest.approx(100.0)
    assert rep["passed"] is True


# ---- 4. report persistence -------------------------------------------

def test_save_report_writes_json_and_md(tmp_path):
    s, clock = make_session(node_id="simpang_01", scenario="Live RTSP Validation")
    s.apply_payload(payload({"simpang_01": node(0, 0, 0)}))
    s.start_session()
    clock[0] += 125
    s.apply_payload(payload({"simpang_01": node(4, 2, 1)}))
    s.counts.update({"motor": 4, "mobil": 2, "bis/truk": 1})
    s.stop_session()

    path = s.save_report(tmp_path, stamp="20261109_120000", quiet=True)
    assert path.name == "benchmark_simpang_01_20261109_120000.json"
    md = path.with_suffix(".md")
    assert path.exists() and md.exists()

    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("timestamp", "scenario", "node_id", "node_name",
                "duration_sec", "api_connected", "manual_counts",
                "ai_baseline", "ai_final", "ai_delta", "report"):
        assert key in data, key
    assert data["node_id"] == "simpang_01" and data["node_name"] == "Simpang 01"
    assert data["duration_sec"] == pytest.approx(125)
    assert data["api_connected"] is True
    assert data["manual_counts"] == {"motor": 4, "mobil": 2, "bis/truk": 1}
    assert data["ai_delta"] == {"motor": 4, "mobil": 2, "truk_bus": 1}
    assert data["report"]["weighted_accuracy"] == pytest.approx(100.0)

    text = md.read_text(encoding="utf-8")
    assert text.startswith("# Benchmark - Simpang 01")
    assert "Node: `simpang_01`" in text
    assert "Duration: 02:05" in text
    assert "| Motor |" in text or "Motor" in text
    assert "|" in text and "Overall Weighted Accuracy" in text

    # unchanged state -> no duplicate auto-save; new count -> auto-save
    assert s.maybe_save_on_exit(tmp_path) is None
    s.increment("motor")
    saved = s.maybe_save_on_exit(tmp_path)
    assert saved is not None and saved.exists()
    assert saved.with_suffix(".md").exists()

    assert s.maybe_save_on_exit(tmp_path) is None   # fresh session: nothing


def test_no_autosave_when_nothing_counted(tmp_path):
    s, _ = make_session()
    assert s.maybe_save_on_exit(tmp_path) is None
    assert list(tmp_path.iterdir()) == []


# ---- 5. headless render ------------------------------------------------

@pytest.mark.parametrize("state", ["IDLE", "RUNNING", "STOPPED"])
def test_render_headless_returns_bgr_frame(state):
    s, _ = make_session(node_id="simpang_01")
    s.apply_payload(payload({"simpang_01": node(7, 3, 2)}))
    if state != "IDLE":
        s.start_session()
        s.counts.update({"motor": 7, "mobil": 3, "bis/truk": 2})
    if state == "STOPPED":
        s.stop_session()
    assert s.state == state

    img = render(s)
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    assert img.shape == (H, W, 3)
    assert img.any()                       # not a blank frame
    assert mc.UI_REGIONS                   # clickable regions registered

    actions = {a for _, a, _ in mc.UI_REGIONS}
    assert {"inc", "dec", "toggle", "save", "reset", "cycle"} <= actions


def test_render_offline_shows_standalone_and_still_draws():
    s, _ = make_session()
    img = render(s)
    assert img.shape == (H, W, 3) and img.dtype == np.uint8
    assert s.status_label() == "IDLE - Press SPACE"


def test_render_status_labels():
    s, clock = make_session()
    assert s.status_label() == "IDLE - Press SPACE"
    s.start_session()
    clock[0] += 84
    assert s.status_label() == "REC 01:24"
    s.stop_session()
    assert s.status_label() == "STOPPED 01:24"


# ---- 6. offline / failure resilience ------------------------------------

def test_poll_once_survives_network_failure(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(mc, "fetch_traffic_data", boom)
    s, _ = make_session(node_id="simpang_01")
    assert s.poll_once() is False
    assert s.api_connected is False
    assert "connection refused" in s.api_error

    # counter still fully usable standalone
    s.increment("motor")
    s.start_session()
    rep = s.live_report()
    assert rep["gt_total"] == 1 and rep["pred_total"] == 0
    img = render(s)
    assert img.shape == (H, W, 3)


def test_poll_once_survives_bad_payload(monkeypatch):
    monkeypatch.setattr(mc, "fetch_traffic_data", lambda *a, **k: {"nope": 1})
    s, _ = make_session()
    assert s.apply_payload({"nope": 1}) is False
    assert s.node_id is None and s.ai_latest == mc.zeros()
    assert s.poll_once() is False
    s.increment("mobil")                   # still counts
    assert s.counts["mobil"] == 1


def test_poll_once_rejects_unreachable_server():
    s = BenchmarkSession(api_url="http://127.0.0.1:9", node_id="x")
    assert s.poll_once(timeout=0.5) is False
    assert s.api_connected is False


# ---- shortcuts & CLI (regression guard) --------------------------------

def test_handle_key_shortcuts(tmp_path):
    s, _ = make_session()
    assert handle_key(s, ord("q"), tmp_path) is True and s.counts["motor"] == 1
    assert handle_key(s, ord("w"), tmp_path) is True and s.counts["mobil"] == 1
    assert handle_key(s, ord("l"), tmp_path) is True and s.counts["bis/truk"] == 1
    assert handle_key(s, ord("Q"), tmp_path) is True and s.counts["motor"] == 0
    assert handle_key(s, ord("W"), tmp_path) is True and s.counts["mobil"] == 0
    assert handle_key(s, ord("L"), tmp_path) is True and s.counts["bis/truk"] == 0
    assert handle_key(s, ord("Q"), tmp_path) is True and s.counts["motor"] == 0

    handle_key(s, 32, tmp_path)
    assert s.state == "RUNNING"
    handle_key(s, 32, tmp_path)
    assert s.state == "STOPPED"
    handle_key(s, ord("r"), tmp_path)
    assert s.state == "IDLE"
    assert handle_key(s, 27, tmp_path) is False     # ESC exits


def test_click_regions_dispatch(tmp_path):
    s, _ = make_session()
    render(s)
    plus = next(r for r in mc.UI_REGIONS if r[1] == "inc" and r[2] == "motor")
    x0, y0, _, _ = plus[0]
    assert mc.handle_click(s, x0 + 2, y0 + 2, tmp_path) == "inc"
    assert s.counts["motor"] == 1
    assert mc.handle_click(s, 0, H - 1, tmp_path) is None


def test_parse_args_defaults_and_overrides():
    a = parse_args([])
    assert a.api_url == "http://127.0.0.1:8000"
    assert a.node is None
    assert a.scenario == "Live RTSP Validation"
    assert a.output_dir == "logs/benchmarks"

    b = parse_args(["--api-url", "http://x:1", "--node", "n1",
                    "--scenario", "S", "--output-dir", "out"])
    assert (b.api_url, b.node, b.scenario, b.output_dir) == \
        ("http://x:1", "n1", "S", "out")
