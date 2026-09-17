/* Smart ATCS Banyumas — Multi-Node Command Center
   Leaflet map + glass panel fed by the multi-node FastAPI backend.
   Markers come from /api/v1/traffic-data; clicking one (or a list button)
   selects the node shown in the sidebar and on the MJPEG <img>. */

"use strict";

const API_BASE = "http://localhost:8000";
const POLL_MS = 2000;

const STATUSES = ["LANCAR", "PADAT", "MACET"];

const el = {
  motor: document.getElementById("count-motor"),
  mobil: document.getElementById("count-mobil"),
  truk: document.getElementById("count-truk"),
  statusText: document.getElementById("status-text"),
  occupancy: document.getElementById("occupancy-value"),
  intersection: document.getElementById("intersection-name"),
  fps: document.getElementById("fps-value"),
  conn: document.getElementById("conn-indicator"),
  connText: document.getElementById("conn-text"),
  clock: document.getElementById("clock"),
  video: document.getElementById("video-feed"),
  camTag: document.getElementById("cam-tag"),
  nodeList: document.getElementById("node-list"),
};

let map = null;
const markers = {};   // node_id -> {marker, status, online}
const knownNodes = {}; // node_id -> latest JSON for selection rendering
let selectedId = null;

/* ---- Map ------------------------------------------------------ */

function initMap() {
  map = L.map("map", {
    center: [-7.4245, 109.2302],
    zoom: 13,
    zoomControl: false,
  });
  L.control.zoom({ position: "topright" }).addTo(map);

  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    maxZoom: 19,
  }).addTo(map);
}

function makeIcon(nodeId, name, status) {
  return L.divIcon({
    className: `veh-marker st-${status}`,
    html: '<div class="pin"></div><div class="pin-label">' +
          escapeHtml(name) + "</div>",
    iconSize: [22, 22],
    iconAnchor: [11, 22],
  });
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;",
    '"': "&quot;", "'": "&#39;",
  })[c]);
}

/* ---- Nodes ---------------------------------------------------- */

function upsertMarker(nodeId, node) {
  if (typeof node.lat !== "number" || typeof node.lng !== "number") return;
  const entry = markers[nodeId];
  if (!entry) {
    const marker = L.marker([node.lat, node.lng], {
      icon: makeIcon(nodeId, node.nama, "unknown"),
      title: node.nama,
    }).addTo(map);
    marker.on("click", () => selectNode(nodeId));
    marker.bindTooltip(node.nama, { direction: "top", offset: [0, -14] });
    markers[nodeId] = { marker, status: "unknown", online: true };
  }
  const st = markers[nodeId];
  // Recreate icon only when visible state changes (avoids flicker).
  const want = node.online
    ? (STATUSES.includes(node.status) ? node.status : "unknown")
    : "DISCONNECTED";
  if (st.status !== want) {
    st.marker.setIcon(makeIcon(nodeId, node.nama, want));
    st.status = want;
  }
}

function buildNodeList(nodes) {
  // Buttons appear in first-seen order; only added, never re-sorted.
  for (const [nodeId, node] of Object.entries(nodes)) {
    if (document.getElementById(`nodebtn-${nodeId}`)) continue;
    const btn = document.createElement("button");
    btn.id = `nodebtn-${nodeId}`;
    btn.type = "button";
    btn.className = "node-btn";
    btn.addEventListener("click", () => {
      selectNode(nodeId);
      const m = markers[nodeId];
      if (m) map.flyTo(m.marker.getLatLng(), 15, { duration: 0.8 });
    });
    el.nodeList.appendChild(btn);
  }
}

function refreshNodeList(nodes) {
  for (const [nodeId, node] of Object.entries(nodes)) {
    const btn = document.getElementById(`nodebtn-${nodeId}`);
    if (!btn) continue;
    btn.dataset.status = node.online ? node.status : "OFFLINE";
    btn.classList.toggle("active", nodeId === selectedId);
    btn.innerHTML = '<span class="node-dot"></span>' +
      escapeHtml(node.nama);
  }
}

/* ---- Selection ------------------------------------------------ */

function selectNode(nodeId) {
  selectedId = nodeId;
  const node = knownNodes[nodeId];
  el.video.src = `${API_BASE}/api/v1/video-feed/${nodeId}`;
  if (node) renderNode(node);
}

function renderNode(node) {
  el.motor.textContent = node.motor ?? 0;
  el.mobil.textContent = node.mobil ?? 0;
  el.truk.textContent = node.truk ?? 0;
  el.occupancy.textContent = node.occupancy ?? "--";
  el.fps.textContent = (typeof node.fps === "number"
    ? node.fps.toFixed(1)
    : "--");
  el.intersection.textContent = node.nama ?? "--";
  el.camTag.textContent = `CAM · ${node.nama ?? "--"}`;
  const known = STATUSES.includes(node.status);
  document.body.dataset.status = node.online
    ? (known ? node.status : "unknown")
    : "DISCONNECTED";
  el.statusText.textContent = node.online
    ? (known ? node.status : "MENGHUBUNGKAN…")
    : "OFFLINE";
}

/* ---- Connection ----------------------------------------------- */

function setDisconnected() {
  document.body.classList.add("disconnected");
  el.conn.classList.remove("online");
  el.conn.classList.add("offline");
  el.connText.textContent = "Terputus dari server";
  el.fps.textContent = "--";
}

function setConnected() {
  document.body.classList.remove("disconnected");
  el.conn.classList.add("online");
  el.conn.classList.remove("offline");
  el.connText.textContent = "Terhubung";
}

/* ---- Polling -------------------------------------------------- */

async function poll() {
  try {
    const res = await fetch(`${API_BASE}/api/v1/traffic-data`, {
      cache: "no-store",
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const nodes = data.nodes ?? {};
    Object.assign(knownNodes, nodes);
    for (const [nodeId, node] of Object.entries(nodes)) {
      upsertMarker(nodeId, node);
    }
    buildNodeList(nodes);
    refreshNodeList(nodes);
    if (!selectedId || !nodes[selectedId]) {
      const first = Object.keys(nodes)[0];
      if (first) selectNode(first);
    } else {
      renderNode(nodes[selectedId]);
      refreshNodeList(nodes);
    }
    setConnected();
  } catch (err) {
    console.warn("[ATCS] poll failed:", err.message);
    setDisconnected();
  }
}

/* ---- Clock ---------------------------------------------------- */

function tickClock() {
  el.clock.textContent = new Date().toLocaleTimeString("id-ID", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/* ---- Boot ----------------------------------------------------- */

initMap();
setConnected(); // optimistic; first failed poll flips it back
poll();
setInterval(poll, POLL_MS);
tickClock();
setInterval(tickClock, 1000);
