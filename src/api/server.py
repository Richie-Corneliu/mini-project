"""Multi-node API bridge: FastAPI serves tracker state + MJPEG streams
for the Command Center WebGIS. One worker thread per CCTV node calls the
setters below; the frontend polls /api/v1/traffic-data and opens
/api/v1/video-feed/{node_id} per selected node."""
import os
import threading
import time
from pathlib import Path

import csv
import io

import yaml
from dotenv import load_dotenv
from typing import Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from src.utils.stats_store import today_wib
import cv2

# Credentials live in .env (gitignored), never in nodes.yaml. Loading here
# covers every entry point that imports the server; main.py also imports it.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

app = FastAPI(
    title="Smart ATCS Dishub Banyumas - Multi-Node API",
    description="Backend API streaming & analytics data untuk Command Center WebGIS",
    version="2.0.0"
)

# Izinkan CORS agar WebGIS (HTML/JS) bisa mengakses API tanpa diblokir browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Status dianggap OFFLINE bila worker tidak update lebih dari ini (detik)
STALE_SEC = 10.0

_lock = threading.Lock()
_command_lock = threading.Lock()

# Pending operator commands per node. Workers consume commands atomically.
COMMAND_QUEUE = {}


def _expand_source(source):
    """Fill {user}/{pass} placeholders in a stream URL from the environment.

    nodes.yaml carries rtsp://{user}:{pass}@host/... so no credential is ever
    committed. A URL without placeholders (HLS, USB index) passes through
    untouched; a placeholder with no matching env var stays literal, which
    makes the failure visible in the logs instead of silently connecting
    with an empty password.
    """
    if not isinstance(source, str):
        return source
    return (source.replace("{user}", os.getenv("CCTV_USER", "{user}"))
                  .replace("{pass}", os.getenv("CCTV_PASS", "{pass}")))


def load_node_registry(path=None):
    """Daftar node dari config/nodes.yaml: [{id, nama, lat, lng, source}]."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config" / "nodes.yaml"
    try:
        with open(path, encoding="utf-8") as f:
            nodes = yaml.safe_load(f).get("nodes", []) or []
    except FileNotFoundError:
        return []
    for node in nodes:
        if "source" in node:
            node["source"] = _expand_source(node["source"])
    return nodes


def _blank_state():
    return {"motor": 0, "mobil": 0, "truk": 0, "occupancy": 0,
            "status": "LANCAR", "fps": 0.0, "last_seen": 0.0,
            "start_time": 0.0, "peak_occupancy": 0, "hourly_counts": {},
            "reconnecting": False}


def _registry():
    """nodes.yaml dibaca segar tiap request: node baru muncul tanpa restart."""
    return {n["id"]: n for n in load_node_registry()}


# Global State per node (ditulis worker AI, dibaca endpoint JSON)
nodes_state = {}

# Global frame storage per node untuk Video Streaming MJPEG
latest_frames = {}

# Node yang sedang ditonton operator (di-set lewat POST /api/v1/set-focus).
# Dibaca worker AI di main.py: hanya node ini yang menjalankan render + encode
# MJPEG penuh, sisanya melewatkannya agar GPU/CPU tidak bekerja untuk stream
# yang tidak dilihat.
ACTIVE_FOCUS_NODE = None
_focus_lock = threading.Lock()


def _status_for(occupancy: int) -> str:
    if occupancy >= 65:
        return "MACET"
    if occupancy >= 20:
        return "PADAT"
    return "LANCAR"


def _clean_hourly(raw):
    """Normalize hourly buckets to {hour: {motor, mobil, truk, total}}.

    The worker sends per-class buckets, but a bucket can legitimately carry a
    total with no class split yet (legacy snapshot, or an hour the worker
    attributed to an unknown label). That total is preserved rather than
    recomputed as 0, otherwise the frontend gets a 693-vehicle hour with three
    empty segments and draws nothing.
    """
    out = {}
    for hour, bucket in (raw or {}).items():
        if isinstance(bucket, dict):
            b = {k: int(bucket.get(k, 0)) for k in ("motor", "mobil", "truk")}
            b["total"] = max(int(bucket.get("total") or 0), sum(b.values()))
        else:
            b = {"motor": 0, "mobil": 0, "truk": 0, "total": int(bucket or 0)}
        out[str(hour)] = b
    return out


def update_node_data(node_id: str, data: dict):
    """Dipanggil worker AI tiap frame. data: motor/mobil/truk/occupancy/fps."""
    with _lock:
        st = nodes_state.setdefault(node_id, _blank_state())
        st["motor"] = int(data.get("motor", 0))
        st["mobil"] = int(data.get("mobil", 0))
        st["truk"] = int(data.get("truk", 0))
        st["occupancy"] = int(data.get("occupancy", 0))
        st["fps"] = round(float(data.get("fps", 0.0)), 1)
        st["start_time"] = float(data.get("start_time", st["start_time"]))
        st["peak_occupancy"] = int(data.get("peak_occupancy", st["peak_occupancy"]))
        st["hourly_counts"] = _clean_hourly(
            data.get("hourly_counts", st["hourly_counts"]))
        st["status"] = _status_for(st["occupancy"])
        st["last_seen"] = time.monotonic()
        # A frame just arrived: the stream is healthy again.
        st["reconnecting"] = False


# mark_node_reconnecting may be called every poll/frame while a stream stalls;
# remember the last write per node so it does not contend on _lock each time.
RECONNECT_MARK_SEC = 1.0
_last_reconnect_mark = {}


def mark_node_reconnecting(node_id: str, min_interval=RECONNECT_MARK_SEC) -> bool:
    """Flag a node's stream as stalled/reconnecting until the next frame.

    Throttled per node: the caller can invoke this on every empty read without
    taking the state lock more than once a second. Returns True when the flag
    was actually written.
    """
    now = time.monotonic()
    with _lock:
        last = _last_reconnect_mark.get(node_id)
        if last is not None and now - last < min_interval:
            return False
        _last_reconnect_mark[node_id] = now
        st = nodes_state.setdefault(node_id, _blank_state())
        st["reconnecting"] = True
    return True


def stream_health(st, now=None) -> tuple:
    """(stream_status, stale_seconds) for one node state snapshot.

    OFFLINE   never produced a frame (last_seen == 0)
    RECONNECTING  no frame for > STALE_SEC, or the worker flagged a stall
    LIVE      fresh frame within STALE_SEC
    stale_seconds is None before the first frame.
    """
    if now is None:
        now = time.monotonic()
    last_seen = float(st.get("last_seen") or 0.0)
    if last_seen <= 0.0:
        return "OFFLINE", None
    stale = max(0.0, now - last_seen)
    if st.get("reconnecting") or stale > STALE_SEC:
        return "RECONNECTING", round(stale, 1)
    return "LIVE", round(stale, 1)


def seed_node_data(node_id: str, data: dict):
    """Isi state node dari snapshot harian sebelum frame pertama datang.

    Dipakai saat startup: kamera bisa sedang offline / warming up, dashboard
    tetap menampilkan total dan grafik jam yang dipulihkan. `last_seen`
    dikosongkan supaya node jujur berstatus OFFLINE sampai worker melapor.
    """
    update_node_data(node_id, data)
    with _lock:
        nodes_state[node_id]["last_seen"] = 0.0


def pop_command(node_id: str):
    """Return and remove the next pending command for a node."""
    with _command_lock:
        commands = COMMAND_QUEUE.get(node_id)
        if not commands:
            return None
        command = commands.pop(0)
        if not commands:
            COMMAND_QUEUE.pop(node_id, None)
        return command


def clear_commands(node_id: str) -> None:
    with _command_lock:
        COMMAND_QUEUE.pop(node_id, None)


def set_latest_frame(node_id: str, frame):
    """Dipanggil worker AI tiap frame untuk streaming MJPEG node tersebut."""
    ret, jpeg = cv2.imencode('.jpg', frame)
    if ret:
        with _lock:
            latest_frames[node_id] = jpeg.tobytes()


def get_active_focus():
    """Node id yang sedang ditonton operator, atau None saat kembali ke peta.

    Dibaca worker AI (main.py) tiap iterasi untuk memilih jalur render:
    node fokus menggambar overlay + encode MJPEG, sisanya melewatinya.
    """
    with _focus_lock:
        return ACTIVE_FOCUS_NODE


def _public_state(node_id, reg, st, now):
    last_seen = st.get("last_seen", 0.0)
    online = last_seen > 0 and (now - last_seen) <= STALE_SEC
    stream_status, stale_seconds = stream_health(st, now)
    return {
        "nama": reg.get("nama", node_id),
        "lat": reg.get("lat"),
        "lng": reg.get("lng"),
        "status": st["status"] if online else "OFFLINE",
        "stream_status": stream_status,
        "stale_seconds": stale_seconds,
        "motor": st["motor"],
        "mobil": st["mobil"],
        "truk": st["truk"],
        "occupancy": st["occupancy"],
        "fps": st["fps"],
        "start_time": st["start_time"],
        "peak_occupancy": st["peak_occupancy"],
        "hourly_counts": {h: dict(b) for h, b in st["hourly_counts"].items()},
        "online": online,
    }


# -------------------------------------------------------------------
# ENDPOINTS API
# -------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "Smart ATCS API Server Running", "status": "active",
            "nodes": list(_registry())}


@app.get("/api/v1/nodes")
def get_nodes():
    """Identitas node dari nodes.yaml, dibaca segar tiap request.

    Frontend memakai ini untuk membangun marker + sidebar secara dinamis;
    file yang diedit langsung terlihat tanpa restart server.
    """
    reg = _registry()
    return {"nodes": [
        {"id": nid, "nama": reg.get("nama", nid),
         "lat": reg.get("lat"), "lng": reg.get("lng")}
        for nid, reg in reg.items()
    ]}


@app.get("/api/v1/traffic-data")
def get_traffic_data():
    """Endpoint JSON untuk ditarik oleh Leaflet.js / app.js"""
    reg = _registry()
    now = time.monotonic()
    with _lock:
        tracked = dict(nodes_state)
    snap = {nid: _public_state(nid, reg.get(nid, {}), st, now)
            for nid, st in tracked.items()}
    for nid, reg_node in reg.items():
        snap.setdefault(nid, _public_state(nid, reg_node,
                                           tracked.get(nid, _blank_state()), now))
    return {"nodes": snap}


@app.post("/api/v1/set-focus/{node_id}")
def set_focus(node_id: str):
    """Tandai satu node sebagai fokus live; worker-nya merender + encode MJPEG.

    Frontend memanggil ini saat membuka Counting View. `none` mereset fokus
    (kembali ke peta) sehingga semua node melewatkan render.
    """
    global ACTIVE_FOCUS_NODE
    with _focus_lock:
        ACTIVE_FOCUS_NODE = None if node_id == "none" else node_id
        current = ACTIVE_FOCUS_NODE
    return {"focus": current}


@app.post("/api/v1/control/{node_id}/{action}")
def control_node(node_id: str, action: str,
                 payload: Optional[dict] = Body(default=None)):
    if node_id not in _registry() and node_id not in nodes_state:
        raise HTTPException(status_code=404, detail=f"unknown node {node_id}")
    if action == "update_zone":
        # Trust boundary: payload comes from the browser, so every point is
        # re-checked as a 0.0-1.0 pair before it reaches the AI worker.
        points = (payload or {}).get("points")
        try:
            clean = [[float(p[0]), float(p[1])] for p in points]
        except (TypeError, ValueError, IndexError):
            clean = None
        if (not isinstance(points, list) or len(points) < 3
                or clean is None
                or not all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in clean)):
            raise HTTPException(
                status_code=400,
                detail="update_zone needs points: [[x, y], ...] with >=3 "
                       "pairs normalized to 0.0-1.0")
        command = {"action": "update_zone", "points": clean}
    elif action in {"reset", "record"}:
        command = action
    else:
        raise HTTPException(status_code=400,
                            detail="unsupported action; use reset, record "
                                   "or update_zone")
    with _command_lock:
        COMMAND_QUEUE.setdefault(node_id, []).append(command)
    return {"node_id": node_id, "action": action, "queued": True}


def generate_video_stream(node_id: str):
    while True:
        with _lock:
            jpg = latest_frames.get(node_id)
        if jpg is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
        time.sleep(0.04)  # ~25 FPS stream


@app.get("/api/v1/nodes/{node_id}/export-csv")
def export_node_csv(node_id: str):
    """Laporan harian CSV per simpang: 24 baris jam + baris TOTAL HARIAN.

    Snapshot nodes_state diambil di bawah _lock (deep-copy) supaya tidak
    ada race dengan thread worker AI yang menulis state di frame berikutnya.
    BOM UTF-8 disisipkan agar Excel Windows langsung menampilkan kolom rapi.
    """
    with _lock:
        if node_id not in nodes_state:
            raise HTTPException(status_code=404, detail="Node not found")
        st = dict(nodes_state[node_id])
        st["hourly_counts"] = {h: dict(b) for h, b in st["hourly_counts"].items()}

    reg = _registry().get(node_id, {})
    nama = reg.get("nama", node_id)
    date_wib = today_wib()

    buf = io.StringIO()
    buf.write("\ufeff")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["Tanggal", "ID Simpang", "Nama Simpang", "Rentang Jam",
                     "Motor", "Mobil", "Truk/Bus", "Total Kendaraan",
                     "Status Lalu Lintas"])

    for hour in range(24):
        key = f"{hour:02d}"
        bucket = st["hourly_counts"].get(key, {})
        motor = int(bucket.get("motor", 0))
        mobil = int(bucket.get("mobil", 0))
        truk = int(bucket.get("truk", 0))
        total = max(int(bucket.get("total", 0)), motor + mobil + truk)
        writer.writerow([date_wib, node_id, nama,
                         f"{key}:00 - {key}:59",
                         motor, mobil, truk, total, st["status"]])

    motor = int(st.get("motor", 0))
    mobil = int(st.get("mobil", 0))
    truk = int(st.get("truk", 0))
    total = motor + mobil + truk
    writer.writerow([date_wib, node_id, nama, "00:00 - 23:59 (Total)",
                     motor, mobil, truk, total, st["status"]])

    return Response(
        content=buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition":
                 f'attachment; filename="laporan_lalin_{node_id}_{date_wib}.csv"'},
    )


@app.get("/api/v1/video-feed/{node_id}")
def video_feed(node_id: str):
    """Endpoint Video Stream MJPEG untuk elemen <img> pada WebGIS"""
    if node_id not in _registry() and node_id not in nodes_state:
        raise HTTPException(status_code=404, detail=f"unknown node {node_id}")
    return StreamingResponse(
        generate_video_stream(node_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
