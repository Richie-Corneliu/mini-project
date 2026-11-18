"""Multi-node API bridge: FastAPI serves tracker state + MJPEG streams
for the Command Center WebGIS. One worker thread per CCTV node calls the
setters below; the frontend polls /api/v1/traffic-data and opens
/api/v1/video-feed/{node_id} per selected node."""
import csv
import hmac
import io
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import yaml
from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from src.utils.stats_store import (
    StatsStore,
    all_slot_keys,
    format_slot_interval,
    normalize_node,
    today_wib,
)
from src.utils.zone_store import persist_zone

# Credentials live in .env (gitignored), never in nodes.yaml. Loading here
# covers every entry point that imports the server; main.py also imports it.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

app = FastAPI(
    title="Smart ATCS Dishub Banyumas - Multi-Node API",
    description="Backend API streaming & analytics data untuk Command Center WebGIS",
    version="2.0.0"
)


def _cors_origins() -> list:
    raw = os.getenv("ATCS_CORS_ORIGINS", "")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["*"]


def _require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    expected = os.getenv("ATCS_API_KEY", "").strip()
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


# Izinkan CORS agar WebGIS (HTML/JS) bisa mengakses API tanpa diblokir browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Status dianggap OFFLINE bila worker tidak update lebih dari ini (detik)
STALE_SEC = 10.0

_lock = threading.Lock()
_command_lock = threading.Lock()
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_stats_store = None


def register_stats_store(store) -> None:
    """Bind the active StatsStore so historical dates and archives can be queried."""
    global _stats_store
    with _lock:
        _stats_store = store


def get_stats_store() -> StatsStore:
    with _lock:
        store = _stats_store
    return store if store is not None else StatsStore()

# Pending operator commands per node. Workers consume commands atomically.
COMMAND_QUEUE = {}
_node_frame_sizes = {}
_zone_update_hook = None
_zone_store_path = None
app.state.zone_update_hook = None
app.state.apply_node_zones = None


def register_zone_update_hook(hook, nodes_path=None) -> None:
    """Register live zone-update callback (apply_node_zones) on the API server."""
    global _zone_update_hook, _zone_store_path
    with _lock:
        _zone_update_hook = hook
        _zone_store_path = Path(nodes_path) if nodes_path else None
        app.state.zone_update_hook = hook
        app.state.apply_node_zones = hook


def set_node_frame_size(node_id: str, width: int, height: int) -> None:
    if width > 0 and height > 0:
        with _lock:
            _node_frame_sizes[node_id] = (int(width), int(height))


def get_node_frame_size(node_id: str):
    with _lock:
        return _node_frame_sizes.get(node_id)


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
            "slot_counts": {}, "session_starts": [], "reconnecting": False}


def _registry():
    """nodes.yaml dibaca segar tiap request: node baru muncul tanpa restart."""
    return {n["id"]: n for n in load_node_registry()}


# Global State per node (ditulis worker AI, dibaca endpoint JSON)
nodes_state = {}

# Global frame storage per node untuk Video Streaming MJPEG
latest_frames = {}
_frame_seqs = {}

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


def _clean_buckets(raw):
    """Normalize hour or 15m slot buckets to {key: {motor, mobil, truk, total}}."""
    out = {}
    for key, bucket in (raw or {}).items():
        if isinstance(bucket, dict):
            b = {
                "motor": int(bucket.get("motor", 0)),
                "mobil": int(bucket.get("mobil", 0)),
                "truk": int(bucket.get("truk", bucket.get("truk/bus", 0))),
            }
            b["total"] = max(int(bucket.get("total") or 0), sum(b.values()))
        else:
            b = {"motor": 0, "mobil": 0, "truk": 0, "total": int(bucket or 0)}
        out[str(key)] = b
    return out


_clean_hourly = _clean_buckets
_clean_slots = _clean_buckets


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
        if "slot_counts" in data:
            st["slot_counts"] = _clean_slots(data.get("slot_counts"))
        if "session_starts" in data:
            st["session_starts"] = [str(s) for s in (data.get("session_starts") or [])]
        st["status"] = _status_for(st["occupancy"])
        st["last_seen"] = time.monotonic()
        # A frame just arrived: the stream is healthy again.
        st["reconnecting"] = False
        fw = data.get("frame_width")
        fh = data.get("frame_height")
        if fw and fh:
            _node_frame_sizes[node_id] = (int(fw), int(fh))


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
    if frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 2:
        set_node_frame_size(node_id, int(frame.shape[1]), int(frame.shape[0]))
    ret, jpeg = cv2.imencode('.jpg', frame)
    if ret:
        with _lock:
            latest_frames[node_id] = jpeg.tobytes()
            _frame_seqs[node_id] = _frame_seqs.get(node_id, 0) + 1


def get_active_focus():
    """Node id yang sedang ditonton operator, atau None saat kembali ke peta.

    Dibaca worker AI (main.py) tiap iterasi untuk memilih jalur render:
    node fokus menggambar overlay + encode MJPEG, sisanya melewatinya.
    """
    with _focus_lock:
        return ACTIVE_FOCUS_NODE


def _compute_avg_per_hour(
    total: int,
    slot_counts: dict,
    start_time: float,
    is_historical: bool = False,
    now_wall: Optional[float] = None,
) -> tuple[int, str]:
    if is_historical:
        return round(total / 24.0), "rata-rata 24 jam"
    now = time.time() if now_wall is None else float(now_wall)
    elapsed_hours = max(0.0, (now - start_time) / 3600.0) if start_time > 0 else 0.0
    active_slots = sum(
        1 for b in (slot_counts or {}).values() if int(b.get("total", 0)) > 0
    )
    active_hours = max(elapsed_hours, active_slots * 0.25, 0.25)
    avg = round(total / active_hours) if total > 0 else 0
    return avg, f"dari {active_hours:.1f} jam aktif"


def _public_state(node_id, reg, st, now):
    last_seen = st.get("last_seen", 0.0)
    online = last_seen > 0 and (now - last_seen) <= STALE_SEC
    stream_status, stale_seconds = stream_health(st, now)
    slots = {s: dict(b) for s, b in (st.get("slot_counts") or {}).items()}
    total = int(st["motor"]) + int(st["mobil"]) + int(st["truk"])
    avg_per_hour, avg_basis_label = _compute_avg_per_hour(
        total, slots, float(st.get("start_time") or 0.0), is_historical=False
    )
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
        "slot_counts": slots,
        "session_starts": list(st.get("session_starts") or []),
        "avg_per_hour": avg_per_hour,
        "avg_basis_label": avg_basis_label,
        "online": online,
    }


def _historical_public_state(node_id: str, reg: dict, raw_node: dict) -> dict:
    norm = normalize_node(raw_node)
    counts = norm["counts"]
    total = max(norm["total"], sum(counts.values()))
    hourly = _clean_hourly(norm["hourly_counts"])
    slots = _clean_slots(norm["slot_counts"])
    avg_per_hour, avg_basis_label = _compute_avg_per_hour(
        total, slots, 0.0, is_historical=True
    )
    return {
        "nama": reg.get("nama", node_id),
        "lat": reg.get("lat"),
        "lng": reg.get("lng"),
        "status": "ARSIP",
        "stream_status": "OFFLINE",
        "stale_seconds": None,
        "motor": counts["motor"],
        "mobil": counts["mobil"],
        "truk": counts["truk"],
        "occupancy": 0,
        "fps": 0.0,
        "start_time": 0.0,
        "peak_occupancy": 0,
        "hourly_counts": hourly,
        "slot_counts": slots,
        "session_starts": list(norm["session_starts"]),
        "avg_per_hour": avg_per_hour,
        "avg_basis_label": avg_basis_label,
        "online": False,
        "is_historical": True,
    }


def _validate_date_param(date: Optional[str]) -> Optional[str]:
    if date is None:
        return None
    if not _DATE_RE.match(date):
        raise HTTPException(status_code=400, detail="invalid date format, expected YYYY-MM-DD")
    return date


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
         "lat": reg.get("lat"), "lng": reg.get("lng"),
         "zones": reg.get("zones", [])}
        for nid, reg in reg.items()
    ]}


@app.get("/api/v1/history/dates")
def get_history_dates():
    """Sorted YYYY-MM-DD date strings with recorded counting data."""
    today = today_wib()
    dates = set(get_stats_store().recorded_dates())
    with _lock:
        has_live = any(
            (st["motor"] + st["mobil"] + st["truk"] > 0)
            or bool(st.get("slot_counts"))
            or bool(st.get("hourly_counts"))
            for st in nodes_state.values()
        )
    if has_live:
        dates.add(today)
    return {"dates": sorted(dates), "today": today}


@app.get("/api/v1/history/{node_id}/{date}")
def get_node_history(node_id: str, date: str):
    """Day summary (hourly, 15m slots, session_starts, avg/hour) for one node and date."""
    target_date = _validate_date_param(date)
    reg_map = _registry()
    today = today_wib()
    if target_date == today:
        now = time.monotonic()
        with _lock:
            st = nodes_state.get(node_id)
            if st is not None:
                st = dict(st)
        if st is None:
            if node_id not in reg_map:
                raise HTTPException(status_code=404, detail="Node not found")
            st = _blank_state()
        pub = _public_state(node_id, reg_map.get(node_id, {}), st, now)
        return {"node_id": node_id, "date": target_date, "is_historical": False, **pub}

    doc = get_stats_store().load_date(target_date)
    hist_nodes = (doc or {}).get("nodes", {})
    if node_id not in hist_nodes and node_id not in reg_map:
        raise HTTPException(status_code=404, detail="Node not found")
    pub = _historical_public_state(node_id, reg_map.get(node_id, {}), hist_nodes.get(node_id, {}))
    return {"node_id": node_id, "date": target_date, **pub}


@app.get("/api/v1/traffic-data")
def get_traffic_data(date: Optional[str] = Query(default=None)):
    """Endpoint JSON untuk ditarik oleh Leaflet.js / app.js."""
    target_date = _validate_date_param(date)
    reg = _registry()
    today = today_wib()
    if target_date is None or target_date == today:
        now = time.monotonic()
        with _lock:
            tracked = dict(nodes_state)
        snap = {nid: _public_state(nid, reg.get(nid, {}), st, now)
                for nid, st in tracked.items()}
        for nid, reg_node in reg.items():
            snap.setdefault(nid, _public_state(nid, reg_node,
                                               tracked.get(nid, _blank_state()), now))
        return {"date": today, "is_historical": False, "nodes": snap}

    doc = get_stats_store().load_date(target_date)
    hist_nodes = (doc or {}).get("nodes", {})
    snap = {
        nid: _historical_public_state(nid, reg.get(nid, {}), node_doc)
        for nid, node_doc in hist_nodes.items()
    }
    for nid, reg_node in reg.items():
        snap.setdefault(nid, _historical_public_state(nid, reg_node, hist_nodes.get(nid, {})))
    return {"date": target_date, "is_historical": True, "nodes": snap}


@app.post("/api/v1/set-focus/{node_id}", dependencies=[Depends(_require_api_key)])
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


@app.post("/api/v1/control/{node_id}/{action}", dependencies=[Depends(_require_api_key)])
def control_node(node_id: str, action: str,
                 payload: Optional[dict] = Body(default=None)):
    with _lock:
        hook = (_zone_update_hook
                or getattr(app.state, "apply_node_zones", None)
                or getattr(app.state, "zone_update_hook", None))
        nodes_path = _zone_store_path
    reg = {n["id"]: n for n in load_node_registry(nodes_path)} if nodes_path else _registry()
    if node_id not in reg and node_id not in nodes_state:
        raise HTTPException(status_code=404, detail=f"unknown node {node_id}")
    applied = False
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
        fw = (payload or {}).get("frame_width")
        fh = (payload or {}).get("frame_height")
        if not (fw and fh):
            fsize = get_node_frame_size(node_id)
            if fsize is not None:
                fw, fh = fsize
        if hook is not None and fw and fh:
            pixel_pts = [[int(x * float(fw)), int(y * float(fh))] for x, y in clean]
            hook_zones = [{"name": "zone", "polygon": pixel_pts}]
            try:
                hook_zones = persist_zone(node_id, pixel_pts, path=nodes_path)
            except KeyError:
                pass
            hook(node_id, hook_zones)
            applied = True
            command = {"action": "update_zone", "points": clean, "applied": True}
        else:
            command = {"action": "update_zone", "points": clean}
    elif action in {"reset", "record"}:
        command = action
    else:
        raise HTTPException(status_code=400,
                            detail="unsupported action; use reset, record "
                                   "or update_zone")
    with _command_lock:
        COMMAND_QUEUE.setdefault(node_id, []).append(command)
    return {"node_id": node_id, "action": action, "queued": True, "applied": applied}


def generate_video_stream(node_id: str, keepalive_sec: float = 1.0):
    last_seq = None
    last_bytes = None
    last_sent_at = 0.0
    while True:
        with _lock:
            jpg = latest_frames.get(node_id)
            seq = _frame_seqs.get(node_id)
        now = time.monotonic()
        if jpg is not None:
            changed = (seq != last_seq) if seq is not None else (jpg is not last_bytes)
            if changed or (now - last_sent_at >= keepalive_sec):
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
                last_seq = seq
                last_bytes = jpg
                last_sent_at = now
        time.sleep(0.04)  # ~25 FPS stream


def _resolve_export_state(node_id: str, target_date: str, reg_map: dict) -> dict:
    if target_date == today_wib():
        with _lock:
            if node_id in nodes_state:
                raw = nodes_state[node_id]
                return {
                    "motor": int(raw.get("motor", 0)),
                    "mobil": int(raw.get("mobil", 0)),
                    "truk": int(raw.get("truk", 0)),
                    "slot_counts": {s: dict(b) for s, b in (raw.get("slot_counts") or {}).items()},
                    "session_starts": list(raw.get("session_starts") or []),
                }
        if node_id in reg_map:
            doc = get_stats_store().load_date(target_date)
            doc_nodes = (doc or {}).get("nodes", {})
            if node_id in doc_nodes:
                norm = normalize_node(doc_nodes[node_id])
                return {
                    "motor": norm["counts"]["motor"],
                    "mobil": norm["counts"]["mobil"],
                    "truk": norm["counts"]["truk"],
                    "slot_counts": _clean_slots(norm["slot_counts"]),
                    "session_starts": list(norm["session_starts"]),
                }
            return _blank_state()
        raise HTTPException(status_code=404, detail="Node not found")

    doc = get_stats_store().load_date(target_date)
    doc_nodes = (doc or {}).get("nodes", {})
    if node_id in doc_nodes:
        norm = normalize_node(doc_nodes[node_id])
        return {
            "motor": norm["counts"]["motor"],
            "mobil": norm["counts"]["mobil"],
            "truk": norm["counts"]["truk"],
            "slot_counts": _clean_slots(norm["slot_counts"]),
            "session_starts": list(norm["session_starts"]),
        }
    if node_id in reg_map:
        return _blank_state()
    raise HTTPException(status_code=404, detail="Node not found")


@app.get("/api/v1/export-csv/{node_id}")
@app.get("/api/v1/nodes/{node_id}/export-csv")
def export_node_csv(node_id: str, date: Optional[str] = Query(default=None)):
    """Laporan CSV interval 15 menit per simpang (96 slot + baris TOTAL).

    BOM UTF-8 disisipkan agar Excel Windows langsung menampilkan kolom rapi.
    """
    target_date = _validate_date_param(date) or today_wib()
    reg_map = _registry()
    st = _resolve_export_state(node_id, target_date, reg_map)

    buf = io.StringIO()
    buf.write("\ufeff")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["Interval Waktu", "Motor", "Mobil", "Truk/Bus", "Total"])

    slots = st.get("slot_counts") or {}
    starts = st.get("session_starts") or []
    slot_m = slot_b = slot_t = slot_tot = 0
    for skey in all_slot_keys():
        bucket = slots.get(skey, {})
        motor = int(bucket.get("motor", 0))
        mobil = int(bucket.get("mobil", 0))
        truk = int(bucket.get("truk", 0))
        tot = max(int(bucket.get("total", 0)), motor + mobil + truk)
        slot_m += motor
        slot_b += mobil
        slot_t += truk
        slot_tot += tot
        writer.writerow([format_slot_interval(skey, starts), motor, mobil, truk, tot])

    tm = max(slot_m, int(st.get("motor", 0))) if slots else int(st.get("motor", 0))
    tb = max(slot_b, int(st.get("mobil", 0))) if slots else int(st.get("mobil", 0))
    tt = max(slot_t, int(st.get("truk", 0))) if slots else int(st.get("truk", 0))
    grand = max(slot_tot, tm + tb + tt)
    writer.writerow(["TOTAL", tm, tb, tt, grand])

    return Response(
        content=buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition":
                 f'attachment; filename="laporan_lalin_{node_id}_{target_date}.csv"'},
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
