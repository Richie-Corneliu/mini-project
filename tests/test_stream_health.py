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
