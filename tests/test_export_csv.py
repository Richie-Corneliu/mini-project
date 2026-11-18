"""CSV export endpoint checks for GET /api/v1/export-csv/{id} and legacy alias."""
import csv
import io
import json

from fastapi.testclient import TestClient

from src.api.server import app, register_stats_store, seed_node_data
from src.utils.stats_store import StatsStore, today_wib

client = TestClient(app)

HOURLY = {"08": {"motor": 4, "mobil": 2, "truk": 1, "total": 7},
          "09": {"motor": 1, "mobil": 0, "truk": 0, "total": 1}}
SLOTS = {
    "08:00": {"motor": 3, "mobil": 1, "truk": 1, "total": 5},
    "08:15": {"motor": 1, "mobil": 1, "truk": 0, "total": 2},
    "09:30": {"motor": 1, "mobil": 0, "truk": 0, "total": 1},
}


def _rows(text):
    return list(csv.reader(io.StringIO(text.lstrip("\ufeff"))))


def test_export_csv_headers_and_rows():
    seed_node_data("csv_node", {
        "motor": 5, "mobil": 2, "truk": 1,
        "hourly_counts": HOURLY,
        "slot_counts": SLOTS,
        "session_starts": ["08:07"],
    })
    for path in ("/api/v1/export-csv/csv_node", "/api/v1/nodes/csv_node/export-csv"):
        resp = client.get(path)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/csv")
        date_wib = today_wib()
        assert (f'laporan_lalin_csv_node_{date_wib}.csv'
                in resp.headers["content-disposition"])
        assert resp.content.startswith(b"\xef\xbb\xbf")

        rows = _rows(resp.text)
        assert rows[0] == ["Interval Waktu", "Motor", "Mobil", "Truk/Bus", "Total"]
        assert len(rows) == 98  # header + 96 slots (15m) + TOTAL

        slot_rows = rows[1:97]
        assert slot_rows[0] == ["00:00 - 00:15", "0", "0", "0", "0"]
        assert slot_rows[95] == ["23:45 - 00:00", "0", "0", "0", "0"]

        by_label = {r[0]: r[1:] for r in slot_rows}
        assert by_label["08:07 (system start) - 08:15"] == ["3", "1", "1", "5"]
        assert by_label["08:15 - 08:30"] == ["1", "1", "0", "2"]
        assert by_label["09:30 - 09:45"] == ["1", "0", "0", "1"]

        assert rows[97] == ["TOTAL", "5", "2", "1", "8"]


def test_export_csv_unknown_node_returns_404():
    resp = client.get("/api/v1/export-csv/no_such_node")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Node not found"


def test_export_csv_offline_registered_node_returns_zero_csv(monkeypatch):
    import src.api.server as srv

    srv.nodes_state.pop("offline_reg_node", None)
    monkeypatch.setattr(
        srv, "_registry",
        lambda: {"offline_reg_node": {"id": "offline_reg_node", "nama": "Simpang Offline"}},
    )
    resp = client.get("/api/v1/export-csv/offline_reg_node")
    assert resp.status_code == 200
    rows = _rows(resp.text)
    assert len(rows) == 98
    assert rows[1] == ["00:00 - 00:15", "0", "0", "0", "0"]
    assert rows[97] == ["TOTAL", "0", "0", "0", "0"]


def test_export_csv_historical_date_query(tmp_path):
    store = StatsStore(tmp_path / "daily_stats.json", min_persist_uptime_seconds=0.0)
    store.load()
    arch_date = "2026-11-10"
    arch_file = store.archive_path(arch_date)
    arch_file.parent.mkdir(parents=True, exist_ok=True)
    arch_file.write_text(json.dumps({
        "date": arch_date,
        "updated_at": "2026-11-10T23:59:59+07:00",
        "nodes": {
            "hist_node": {
                "total": 12,
                "counts": {"motor": 7, "mobil": 4, "truk": 1},
                "hourly_counts": {"10": {"motor": 7, "mobil": 4, "truk": 1}},
                "slot_counts": {
                    "10:00": {"motor": 4, "mobil": 2, "truk": 1, "total": 7},
                    "10:15": {"motor": 3, "mobil": 2, "truk": 0, "total": 5},
                },
                "session_starts": ["10:10"],
            }
        },
    }), encoding="utf-8")

    register_stats_store(store)
    try:
        resp = client.get(f"/api/v1/export-csv/hist_node?date={arch_date}")
        assert resp.status_code == 200
        assert f"laporan_lalin_hist_node_{arch_date}.csv" in resp.headers["content-disposition"]
        rows = _rows(resp.text)
        assert len(rows) == 98
        by_label = {r[0]: r[1:] for r in rows[1:97]}
        assert by_label["10:10 (system start) - 10:15"] == ["4", "2", "1", "7"]
        assert by_label["10:15 - 10:30"] == ["3", "2", "0", "5"]
        assert rows[97] == ["TOTAL", "7", "4", "1", "12"]
    finally:
        register_stats_store(None)
