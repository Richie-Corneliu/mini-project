"""Stream health metadata checks for /api/v1/traffic-data.

Covers the three states the Command Center switches on:
LIVE / RECONNECTING / OFFLINE.

Run: python -m pytest tests/test_stream_health.py
"""
import time

from fastapi.testclient import TestClient

from src.api.server import (STALE_SEC, app, mark_node_reconnecting,
                            nodes_state, seed_node_data, update_node_data)

client = TestClient(app)


def _node(node_id):
    payload = client.get("/api/v1/traffic-data").json()
    return payload["nodes"][node_id]


def _fresh(node_id):
    nodes_state.pop(node_id, None)
    update_node_data(node_id, {"motor": 3, "mobil": 1, "truk": 0,
                               "occupancy": 4, "fps": 25.0})


def test_fresh_frame_reports_live():
    nid = "health_live_test"
    _fresh(nid)
    node = _node(nid)
    assert node["stream_status"] == "LIVE"
    assert node["stale_seconds"] is not None
    assert node["stale_seconds"] <= 1.0


def test_stale_timestamp_reports_reconnecting():
    nid = "health_stale_test"
    _fresh(nid)
    nodes_state[nid]["last_seen"] = time.monotonic() - STALE_SEC - 5
    node = _node(nid)
    assert node["stream_status"] == "RECONNECTING"
    assert node["stale_seconds"] > STALE_SEC


def test_mark_reconnecting_then_frame_restores_live():
    nid = "health_mark_test"
    _fresh(nid)
    assert _node(nid)["stream_status"] == "LIVE"

    assert mark_node_reconnecting(nid) is True
    assert _node(nid)["stream_status"] == "RECONNECTING"

    _fresh(nid)   # a real frame clears the flag
    assert _node(nid)["stream_status"] == "LIVE"


def test_never_seen_node_reports_offline():
    nid = "health_offline_test"
    nodes_state.pop(nid, None)
    seed_node_data(nid, {"motor": 4, "mobil": 2, "truk": 1,
                         "hourly_counts": {}})
    node = _node(nid)
    assert node["stream_status"] == "OFFLINE"
    assert node["stale_seconds"] is None


def test_api_key_enforcement_and_update_zone_hook_and_mjpeg_dedup(monkeypatch, tmp_path):
    import numpy as np
    import src.api.server as srv

    nid = "ctrl_auth_test"
    nodes_file = tmp_path / "nodes.yaml"
    nodes_file.write_text(
        f"nodes:\n- id: {nid}\n  nama: Auth Test\n  zones:\n"
        "  - name: south\n    polygon: [[0, 0], [100, 0], [100, 100], [0, 100]]\n"
        "    allowed_directions: [\"down\"]\n    flow_max_angle_deg: 55\n",
        encoding="utf-8",
    )

    received = []
    srv.set_node_frame_size(nid, 1000, 800)
    srv.register_zone_update_hook(lambda n, z: received.append((n, z)), nodes_path=nodes_file)
    try:
        # Unset ATCS_API_KEY allows local control routes
        monkeypatch.delenv("ATCS_API_KEY", raising=False)
        assert client.post(f"/api/v1/set-focus/{nid}").status_code == 200

        # Configured ATCS_API_KEY enforces X-API-Key on POST routes
        monkeypatch.setenv("ATCS_API_KEY", "top-secret-42")
        assert client.post(f"/api/v1/set-focus/{nid}").status_code == 401
        assert client.post(f"/api/v1/control/{nid}/reset").status_code == 401
        assert client.post(
            f"/api/v1/control/{nid}/reset", headers={"X-API-Key": "wrong"}
        ).status_code == 401

        res = client.post(
            f"/api/v1/control/{nid}/update_zone",
            headers={"X-API-Key": "top-secret-42"},
            json={"points": [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]},
        )
        assert res.status_code == 200 and res.json()["applied"] is True
        assert received and received[-1][0] == nid
        z0 = received[-1][1][0]
        assert z0["polygon"] == [[100, 160], [900, 160], [900, 640], [100, 640]]
        assert z0["allowed_directions"] == ["down"]
        assert z0["flow_max_angle_deg"] == 55

        # MJPEG stream deduplicates identical seq until new frame or keepalive
        clock = [100.0]
        sleeps = {"count": 0}
        monkeypatch.setattr("src.api.server.time.monotonic", lambda: clock[0])

        def advance_sleep(_dt):
            sleeps["count"] += 1
            clock[0] += 0.05

        monkeypatch.setattr("src.api.server.time.sleep", advance_sleep)
        srv.set_latest_frame(nid, np.zeros((4, 4, 3), dtype=np.uint8))
        gen = srv.generate_video_stream(nid, keepalive_sec=1.0)
        first = next(gen)
        assert first.startswith(b"--frame\r\n")
        assert sleeps["count"] == 0

        # New frame advances _frame_seqs -> next yield happens after just 1 loop sleep
        srv.set_latest_frame(nid, np.full((4, 4, 3), 200, dtype=np.uint8))
        second = next(gen)
        assert sleeps["count"] == 1 and second != first

        # Unchanged frame skips intermediate ticks and only emits once keepalive >= 1.0s
        before_idle = sleeps["count"]
        keepalive = next(gen)
        assert sleeps["count"] - before_idle >= 20
        assert keepalive == second
        gen.close()
    finally:
        srv.register_zone_update_hook(None)
        srv.clear_commands(nid)
        srv.set_focus("none")
        with srv._lock:
            srv._node_frame_sizes.pop(nid, None)
            srv.latest_frames.pop(nid, None)
            srv._frame_seqs.pop(nid, None)



def test_history_dates_traffic_data_and_avg_per_hour_basis(tmp_path):
    import json
    import src.api.server as srv
    from src.utils.stats_store import StatsStore, today_wib

    store = StatsStore(tmp_path / "daily_stats.json", min_persist_uptime_seconds=0.0)
    store.load()
    arch_date = "2026-11-10"
    arch_file = store.archive_path(arch_date)
    arch_file.parent.mkdir(parents=True, exist_ok=True)
    arch_file.write_text(json.dumps({
        "date": arch_date,
        "updated_at": "2026-11-10T23:59:59+07:00",
        "nodes": {
            "hist_avg_node": {
                "total": 240,
                "counts": {"motor": 180, "mobil": 48, "truk": 12},
                "hourly_counts": {"10": {"motor": 180, "mobil": 48, "truk": 12}},
                "slot_counts": {
                    "10:00": {"motor": 180, "mobil": 48, "truk": 12, "total": 240},
                },
                "session_starts": ["10:05"],
            }
        },
    }), encoding="utf-8")

    srv.register_stats_store(store)
    nid = "live_avg_node"
    srv.nodes_state.pop(nid, None)
    try:
        # Live partial-hour node: 100 vehicles across two 15-min slots (= 0.5 active h)
        srv.update_node_data(nid, {
            "motor": 60, "mobil": 30, "truk": 10,
            "occupancy": 5, "fps": 25.0,
            "start_time": time.time() - 1800.0,
            "slot_counts": {
                "10:00": {"motor": 30, "mobil": 15, "truk": 5, "total": 50},
                "10:15": {"motor": 30, "mobil": 15, "truk": 5, "total": 50},
            },
            "session_starts": ["10:10"],
        })
        live_node = client.get("/api/v1/traffic-data").json()["nodes"][nid]
        assert live_node["avg_per_hour"] == 200
        assert live_node["avg_basis_label"] == "dari 0.5 jam aktif"
        assert live_node["session_starts"] == ["10:10"]

        dates_payload = client.get("/api/v1/history/dates").json()
        assert arch_date in dates_payload["dates"]
        assert today_wib() in dates_payload["dates"]

        hist_resp = client.get(f"/api/v1/traffic-data?date={arch_date}").json()
        assert hist_resp["is_historical"] is True
        hist_node = hist_resp["nodes"]["hist_avg_node"]
        assert hist_node["avg_per_hour"] == 10
        assert hist_node["avg_basis_label"] == "rata-rata 24 jam"
        assert hist_node["session_starts"] == ["10:05"]
        assert hist_node["slot_counts"]["10:00"]["total"] == 240

        single_hist = client.get(f"/api/v1/history/hist_avg_node/{arch_date}").json()
        assert single_hist["avg_per_hour"] == 10
        assert single_hist["avg_basis_label"] == "rata-rata 24 jam"

        assert client.get("/api/v1/traffic-data?date=not-a-date").status_code == 400
    finally:
        srv.register_stats_store(None)
        srv.nodes_state.pop(nid, None)
