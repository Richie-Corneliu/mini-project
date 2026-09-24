"""Multi-node API bridge: FastAPI serves tracker state + MJPEG streams
for the Command Center WebGIS. One worker thread per CCTV node calls the
setters below; the frontend polls /api/v1/traffic-data and opens
/api/v1/video-feed/{node_id} per selected node."""
import threading
import time
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2

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


def load_node_registry(path=None):
    """Daftar node dari config/nodes.yaml: [{id, nama, lat, lng, source}]."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config" / "nodes.yaml"
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f).get("nodes", []) or []
    except FileNotFoundError:
        return []


REGISTRY = {n["id"]: n for n in load_node_registry()}

# Global State per node (ditulis worker AI, dibaca endpoint JSON)
nodes_state = {
    nid: {"motor": 0, "mobil": 0, "truk": 0, "occupancy": 0,
          "status": "LANCAR", "fps": 0.0, "last_seen": 0.0,
          "start_time": 0.0, "peak_occupancy": 0, "hourly_counts": {}}
    for nid in REGISTRY
}

# Global frame storage per node untuk Video Streaming MJPEG
latest_frames = {}

# Node yang sedang ditonton operator (di-set lewat POST /api/v1/set-focus).
# Dibaca worker AI di main.py: node ini jalan di target_fps penuh, sisanya
# di-throttle agar GPU hanya bekerja untuk stream yang benar-benar dilihat.
ACTIVE_FOCUS_NODE = None
_focus_lock = threading.Lock()


def _status_for(occupancy: int) -> str:
    if occupancy >= 65:
        return "MACET"
    if occupancy >= 20:
        return "PADAT"
    return "LANCAR"


def update_node_data(node_id: str, data: dict):
    """Dipanggil worker AI tiap frame. data: motor/mobil/truk/occupancy/fps."""
    with _lock:
        st = nodes_state.setdefault(node_id, {"motor": 0, "mobil": 0,
                                               "truk": 0, "occupancy": 0,
                                               "status": "LANCAR", "fps": 0.0,
                                               "last_seen": 0.0,
                                               "start_time": 0.0,
                                               "peak_occupancy": 0,
                                               "hourly_counts": {}})
        st["motor"] = int(data.get("motor", 0))
        st["mobil"] = int(data.get("mobil", 0))
        st["truk"] = int(data.get("truk", 0))
        st["occupancy"] = int(data.get("occupancy", 0))
        st["fps"] = round(float(data.get("fps", 0.0)), 1)
        st["start_time"] = float(data.get("start_time", st["start_time"]))
        st["peak_occupancy"] = int(data.get("peak_occupancy", st["peak_occupancy"]))
        st["hourly_counts"] = {
            str(hour): int(count)
            for hour, count in data.get("hourly_counts", st["hourly_counts"]).items()
        }
        st["status"] = _status_for(st["occupancy"])
        st["last_seen"] = time.monotonic()


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

    Dibaca worker AI (main.py) tiap iterasi untuk memilih laju inferensi:
    node fokus di target_fps penuh, sisanya di-throttle ke throttle_fps.
    """
    with _focus_lock:
        return ACTIVE_FOCUS_NODE


def _public_state(node_id, reg, st, now):
    online = (now - st.get("last_seen", 0.0)) <= STALE_SEC
    return {
        "nama": reg.get("nama", node_id),
        "lat": reg.get("lat"),
        "lng": reg.get("lng"),
        "status": st["status"] if online else "OFFLINE",
        "motor": st["motor"],
        "mobil": st["mobil"],
        "truk": st["truk"],
        "occupancy": st["occupancy"],
        "fps": st["fps"],
        "start_time": st["start_time"],
        "peak_occupancy": st["peak_occupancy"],
        "hourly_counts": dict(st["hourly_counts"]),
        "online": online,
    }


# -------------------------------------------------------------------
# ENDPOINTS API
# -------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "Smart ATCS API Server Running", "status": "active",
            "nodes": list(REGISTRY)}


@app.get("/api/v1/traffic-data")
def get_traffic_data():
    """Endpoint JSON untuk ditarik oleh Leaflet.js / script.js"""
    now = time.monotonic()
    with _lock:
        snap = {nid: _public_state(nid, REGISTRY.get(nid, {}), st, now)
                for nid, st in nodes_state.items()}
    for nid, reg in REGISTRY.items():
        snap.setdefault(nid, _public_state(nid, reg, nodes_state.get(nid, {
            "motor": 0, "mobil": 0, "truk": 0, "occupancy": 0,
            "status": "LANCAR", "fps": 0.0, "last_seen": 0.0,
            "start_time": 0.0, "peak_occupancy": 0, "hourly_counts": {}}), now))
    return {"nodes": snap}


@app.post("/api/v1/set-focus/{node_id}")
def set_focus(node_id: str):
    """Tandai satu node sebagai fokus live; worker-nya naik ke target_fps.

    Frontend memanggil ini saat membuka Counting View. `none` mereset fokus
    (kembali ke peta) sehingga semua node kembali di-throttle.
    """
    global ACTIVE_FOCUS_NODE
    with _focus_lock:
        ACTIVE_FOCUS_NODE = None if node_id == "none" else node_id
        current = ACTIVE_FOCUS_NODE
    return {"focus": current}


@app.post("/api/v1/control/{node_id}/{action}")
def control_node(node_id: str, action: str):
    if node_id not in REGISTRY and node_id not in nodes_state:
        raise HTTPException(status_code=404, detail=f"unknown node {node_id}")
    if action not in {"reset", "record"}:
        raise HTTPException(status_code=400,
                            detail="unsupported action; use reset or record")
    with _command_lock:
        COMMAND_QUEUE.setdefault(node_id, []).append(action)
    return {"node_id": node_id, "action": action, "queued": True}


def generate_video_stream(node_id: str):
    while True:
        with _lock:
            jpg = latest_frames.get(node_id)
        if jpg is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
        time.sleep(0.04)  # ~25 FPS stream


@app.get("/api/v1/video-feed/{node_id}")
def video_feed(node_id: str):
    """Endpoint Video Stream MJPEG untuk elemen <img> pada WebGIS"""
    if node_id not in REGISTRY and node_id not in nodes_state:
        raise HTTPException(status_code=404, detail=f"unknown node {node_id}")
    return StreamingResponse(
        generate_video_stream(node_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
