"""CSV export endpoint checks for GET /api/v1/nodes/{id}/export-csv.

Run: python -m pytest tests/test_export_csv.py
"""
import csv
import io

from fastapi.testclient import TestClient

from src.api.server import app, seed_node_data
from src.utils.stats_store import today_wib

client = TestClient(app)

HOURLY = {"08": {"motor": 4, "mobil": 2, "truk": 1, "total": 7},
          "09": {"motor": 1, "mobil": 0, "truk": 0, "total": 1}}


def _rows(text):
    return list(csv.reader(io.StringIO(text.lstrip("\ufeff"))))


def test_export_csv_headers_and_rows():
    seed_node_data("csv_node", {"motor": 5, "mobil": 2, "truk": 1,
                                "hourly_counts": HOURLY})
    resp = client.get("/api/v1/nodes/csv_node/export-csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    date_wib = today_wib()
    assert (f'laporan_lalin_csv_node_{date_wib}.csv'
            in resp.headers["content-disposition"])
    assert resp.content.startswith(b"\xef\xbb\xbf")

    rows = _rows(resp.text)
    assert rows[0] == ["Tanggal", "ID Simpang", "Nama Simpang", "Rentang Jam",
                       "Motor", "Mobil", "Truk/Bus", "Total Kendaraan",
                       "Status Lalu Lintas"]
    assert len(rows) == 26  # header + 24 hourly + TOTAL HARIAN

    hour_rows = rows[1:25]
    assert hour_rows[0][3] == "00:00 - 00:59"
    assert hour_rows[23][3] == "23:00 - 23:59"
    # empty hours default to zero
    assert hour_rows[0][4:8] == ["0", "0", "0", "0"]
    # seeded hours match exactly
    idx08 = next(i for i, r in enumerate(hour_rows) if r[3].startswith("08"))
    assert hour_rows[idx08][4:8] == ["4", "2", "1", "7"]

    total_row = rows[25]
    assert total_row[3] == "00:00 - 23:59 (Total)"
    assert total_row[4:8] == ["5", "2", "1", "8"]


def test_export_csv_unknown_node_returns_404():
    resp = client.get("/api/v1/nodes/no_such_node/export-csv")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Node not found"
