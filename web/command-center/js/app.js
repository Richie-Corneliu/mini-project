/* Smart ATCS Banyumas - Command Center
   Node identity (markers + sidebar) comes from /api/v1/nodes, which reads
   nodes.yaml fresh per request, so a new simpang appears without a redeploy.
   /api/v1/traffic-data then overlays live counts on a 1.5s poll; markers and
   sidebar repaint from the same snapshot.

   Two views share one page:
     view-map    WebGIS overview, Leaflet map plus the floating dashboard menu
     view-count  fixed-height split screen: stream + action bar (left),
                 scrollable analytics sidebar (right)
*/

"use strict";

const MAP_CENTER = [-7.4245, 109.2302];
const MAP_ZOOM = 14;
const TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";

/* Backend: FastAPI serves processed MJPEG + tracker JSON. The frontend never
   touches RTSP/HLS directly, only these HTTP endpoints.

   Absolute by default so the page works from Live Server (localhost:5500) and
   from the backend itself (localhost:8000). Override at deploy time by setting
   window.ATCS_API_BASE before this script loads, or ?api= on the query string
   (handy when the dashboard is served from another host). */
const API_BASE = (
  new URLSearchParams(location.search).get("api") ||
  window.ATCS_API_BASE ||
  "http://localhost:8000"
).replace(/\/$/, "") + "/api/v1";
const POLL_MS = 1500;

const STATUS_ORDER = ["LANCAR", "PADAT", "MACET"];
const STATUS_CLASS = {
  LANCAR: "st-LANCAR", PADAT: "st-PADAT", MACET: "st-MACET",
  OFFLINE: "st-OFFLINE",
};

const NOTES = {
  LANCAR: "Arus lalu lintas lancar, tidak ada penumpukan kendaraan.",
  PADAT: "Volume kendaraan meningkat, pantau pergerakan tiap 5 menit.",
  MACET: "Kepadatan tinggi. Pengaturan simpang perlu dikoordinasikan.",
};

const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 50;

/* ---- DOM refs --------------------------------------------------- */

const byId = (id) => document.getElementById(id);

const el = {
  viewMap: byId("view-map"),
  viewCount: byId("view-count"),
  feed: byId("stage-video"),
  backBtn: byId("back-btn"),
  camChip: byId("cam-chip"),

  clockTime: byId("clock-time"),
  clockDate: byId("clock-date"),

  menuCount: byId("menu-count"),
  menuList: byId("menu-list"),
  totalKendaraan: byId("total-kendaraan"),
  totalSimpang: byId("total-simpang"),
  split: {
    LANCAR: byId("split-LANCAR"),
    PADAT: byId("split-PADAT"),
    MACET: byId("split-MACET"),
  },

  panelPlace: byId("panel-place"),
  resetCounterBtn: byId("reset-counter-btn"),
  toggleRecordBtn: byId("toggle-record-btn"),
  drawZoneBtn: byId("draw-zone-btn"),
  saveZoneBtn: byId("save-zone-btn"),
  controlStatus: byId("control-status"),
  sumTotal: byId("sum-total"),
  sumMotor: byId("sum-motor"),
  sumMobil: byId("sum-mobil"),
  sumTruk: byId("sum-truk"),

  countMotor: byId("count-motor"),
  countMobil: byId("count-mobil"),
  countTruk: byId("count-truk"),
  avgHour: byId("avg-hour"),
  peakHour: byId("peak-hour"),

  chart: byId("chart"),
  chartAxis: byId("chart-axis"),
  gauge: byId("gauge"),
  gaugeFill: byId("gauge-fill"),
  occValue: byId("occ-value"),
  gaugeStatus: byId("gauge-status"),
  occLabel: byId("occ-label"),
  occNote: byId("occ-note"),
};

/* ---- State ------------------------------------------------------ */

const nodes = new Map();
const markers = new Map();
let map = null;
let selectedId = null;
let lastEscapeAt = 0;

const total = (node) => node.motor + node.mobil + node.truk;

/* Raw status for data-status attributes, CSS class for st-* selectors. */
function rawStatus(node) {
  if (!node.online) return "OFFLINE";
  return STATUS_ORDER.includes(node.status) ? node.status : "unknown";
}
function stClass(node) {
  return STATUS_CLASS[rawStatus(node)] || "st-unknown";
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;",
    '"': "&quot;", "'": "&#39;",
  })[c]);
}

/* ---- Map -------------------------------------------------------- */

function initMap() {
  map = L.map("map", { center: MAP_CENTER, zoom: MAP_ZOOM });
  L.tileLayer(TILE_URL, {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);
}
/* 44px hit box keeps the pin tappable; ring, core and label centre inside it. */
function markerIcon(node) {
  return L.divIcon({
    className: "veh-marker " + stClass(node),
    html: '<div class="ring"></div><div class="core"></div><div class="pin-label"></div>',
    iconSize: [44, 44],
    iconAnchor: [22, 22],
  });
}

/* Hover preview of the same numbers the sidebar shows. */
function tooltipHtml(node) {
  const status = rawStatus(node);
  return '<strong>' + escapeHtml(node.nama) + "</strong><br>" +
    '<span class="tip-status ' + stClass(node) + '">' + status + "</span>" +
    ' &middot; okupansi <span class="tip-occ">' + (node.occupancy || 0) + "%</span><br>" +
    "Motor " + node.motor + " &middot; Mobil " + node.mobil +
    " &middot; Truk " + node.truk;
}

/* Marker creation and refresh share one path so a poll can recolor a pin in
   place. The icon is rebuilt only when the status class actually changed
   (keeps the pulse from restarting every cycle); the tooltip content is
   refreshed with every snapshot so occupancy stays live. */
function upsertMarkers() {
  if (!map) return;   // Leaflet unavailable: sidebar still renders below.
  nodes.forEach((node) => {
    if (node.lat == null || node.lng == null) return;

    const status = stClass(node);
    const existing = markers.get(node.id);
    if (!existing) {
      const marker = L.marker([node.lat, node.lng], {
        icon: markerIcon(node),
        riseOnHover: true,
        keyboard: true,
        title: node.nama,
        alt: node.nama,
      });

      /* Leaflet injects divIcon html as markup, so the name goes in as text. */
      marker.on("add", () => {
        const icon = marker.getElement();
        const label = icon && icon.querySelector(".pin-label");
        if (label) label.textContent = node.nama;
      });
      marker.on("click", () => openCounting(node.id));
      marker.bindTooltip(tooltipHtml(node), {
        className: "veh-tip", direction: "top", offset: [0, -14],
      });

      marker.addTo(map);
      marker._status = status;
      markers.set(node.id, marker);
      return;
    }

    existing.setLatLng([node.lat, node.lng]);
    if (existing._status !== status) {
      existing.setIcon(markerIcon(node));
      existing._status = status;
    }
    existing.setTooltipContent(tooltipHtml(node));
  });
}

/* ---- Dashboard menu (VIEW 1) ------------------------------------ */

function renderMenu() {
  const list = [...nodes.values()];
  const grand = list.reduce((sum, node) => sum + total(node), 0);

  el.totalKendaraan.textContent = grand;
  el.totalSimpang.textContent = list.length;
  el.menuCount.textContent = list.length + " titik";

  STATUS_ORDER.forEach((status) => {
    const count = list.filter((node) => rawStatus(node) === status).length;
    const seg = el.split[status];
    seg.dataset.count = count;
    seg.style.flexGrow = count || 0.001;
    seg.title = status + ": " + count + " simpang";
  });

  el.menuList.innerHTML = list.map((node) => (
    '<li><button type="button" class="menu-item" data-id="' + node.id +
    '" data-status="' + rawStatus(node) + '">' +
      '<span class="menu-dot"></span>' +
      '<span class="menu-name">' + escapeHtml(node.nama) + "</span>" +
      '<span class="menu-val">' + total(node) + "</span>" +
    "</button></li>"
  )).join("");
}

el.menuList.addEventListener("click", (event) => {
  const item = event.target.closest(".menu-item");
  if (item) openCounting(item.dataset.id);
});

/* ---- View switching --------------------------------------------- */

function showView(which) {
  const toCounting = which === "count";

  el.viewMap.classList.toggle("is-visible", !toCounting);
  el.viewCount.classList.toggle("is-visible", toCounting);

  /* Leaflet measures wrong while its pane is hidden, so re-measure on return. */
  if (!toCounting && map) map.invalidateSize();

  /* The MJPEG connection lives in the img src, so dropping it on the way out
     is what stops the stream and frees the browser's bandwidth. */
  if (toCounting) attachStream(selectedId);
  else clearStream();
}

/* ---- Backend bridge (focus + MJPEG) ----------------------------- */

/* Ask the worker to prioritise this node's inference. Fire-and-forget: the
   endpoint is optional, so a missing backend must not break the UI. */
function setFocus(nodeId) {
  fetch(API_BASE + "/set-focus/" + nodeId, { method: "POST" }).catch(() => {});
}

async function sendControl(action) {
  if (!selectedId) return;

  el.resetCounterBtn.disabled = true;
  el.toggleRecordBtn.disabled = true;
  el.controlStatus.textContent = "Mengirim perintah...";
  try {
    const res = await fetch(API_BASE + "/control/" + selectedId + "/" + action, {
      method: "POST",
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    el.controlStatus.textContent = action === "reset"
      ? "Perintah reset diterima."
      : "Perintah rekam diterima.";
  } catch (err) {
    el.controlStatus.textContent = "Perintah gagal dikirim.";
  } finally {
    el.resetCounterBtn.disabled = false;
    el.toggleRecordBtn.disabled = false;
  }
}

el.resetCounterBtn.addEventListener("click", () => sendControl("reset"));
el.toggleRecordBtn.addEventListener("click", () => sendControl("record"));

/* ---- Zone drawing (VIEW 2) --------------------------------------- *
 *
 * The MJPEG <img> fills the stage with object-fit: contain, so the stream is
 * letterboxed inside the canvas. Clicks are only meaningful inside the video
 * box, and must be remapped to 0.0-1.0 ratios of the RAW frame before being
 * sent; the backend multiplies by the actual frame dimensions. */

const ZONE_COLORS = { line: "#38BDF8", fill: "rgba(56, 189, 248, 0.18)" };

const zoneDraw = {
  active: false,
  points: [],          // [x, y] in raw-frame ratios (0.0-1.0)
  raf: null,
};

function zoneCanvas() {
  return document.getElementById("zone-canvas");
}

/* Actual displayed video box inside the letterboxing stage. */
function videoBox() {
  const feed = el.feed;
  const stage = feed.parentElement;
  const stageW = stage.clientWidth;
  const stageH = stage.clientHeight;
  const nw = feed.naturalWidth || feed.videoWidth;
  const nh = feed.naturalHeight || feed.videoHeight;
  if (!nw || !nh) return null;
  const scale = Math.min(stageW / nw, stageH / nh);
  const boxW = nw * scale;
  const boxH = nh * scale;
  return { x: (stageW - boxW) / 2, y: (stageH - boxH) / 2, w: boxW, h: boxH };
}

function fitZoneCanvas() {
  const canvas = zoneCanvas();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
}

function drawZonePreview() {
  const canvas = zoneCanvas();
  if (!canvas) return;
  zoneDraw.raf = null;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!zoneDraw.active || zoneDraw.points.length === 0) return;

  const box = videoBox();
  if (!box) return;
  const dpr = window.devicePixelRatio || 1;
  const toPx = (p) => [
    (box.x + p[0] * box.w) * dpr,
    (box.y + p[1] * box.h) * dpr,
  ];

  const pts = zoneDraw.points.map(toPx);
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.lineTo(pts[0][0], pts[0][1]);   // auto-close preview
  ctx.closePath();
  ctx.fillStyle = ZONE_COLORS.fill;
  ctx.fill();
  ctx.strokeStyle = ZONE_COLORS.line;
  ctx.lineWidth = 2 * dpr;
  ctx.stroke();

  for (const [px, py] of pts) {
    ctx.beginPath();
    ctx.arc(px, py, 4 * dpr, 0, Math.PI * 2);
    ctx.fillStyle = ZONE_COLORS.line;
    ctx.fill();
  }
}

function scheduleZonePreview() {
  if (zoneDraw.raf == null) zoneDraw.raf = requestAnimationFrame(drawZonePreview);
}

function startZoneDraw() {
  zoneDraw.active = true;
  zoneDraw.points = [];
  fitZoneCanvas();
  scheduleZonePreview();
  zoneCanvas().style.cursor = "crosshair";
  el.saveZoneBtn.hidden = false;
  el.saveZoneBtn.disabled = true;
  document.getElementById("draw-zone-label").textContent = "Batal Gambar";
  el.controlStatus.textContent = "Klik kanvas untuk menandai titik zona.";
}

function cancelZoneDraw() {
  zoneDraw.active = false;
  zoneDraw.points = [];
  if (zoneDraw.raf != null) { cancelAnimationFrame(zoneDraw.raf); zoneDraw.raf = null; }
  const canvas = zoneCanvas();
  if (canvas) {
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    canvas.style.cursor = "";
  }
  el.saveZoneBtn.hidden = true;
  el.saveZoneBtn.disabled = true;
  document.getElementById("draw-zone-label").textContent = "Gambar Zona";
  el.controlStatus.textContent = "";
}

async function saveZone() {
  if (!selectedId || zoneDraw.points.length < 3) return;
  el.saveZoneBtn.disabled = true;
  el.controlStatus.textContent = "Menyimpan zona...";
  try {
    const res = await fetch(API_BASE + "/control/" + selectedId + "/update_zone", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ points: zoneDraw.points.map((p) => [p[0], p[1]]) }),
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    el.controlStatus.textContent = "Zona baru diterima backend.";
    cancelZoneDraw();
  } catch (err) {
    el.saveZoneBtn.disabled = false;
    el.controlStatus.textContent = "Zona gagal dikirim.";
  }
}

el.drawZoneBtn.addEventListener("click", () => {
  zoneDraw.active ? cancelZoneDraw() : startZoneDraw();
});
el.saveZoneBtn.addEventListener("click", saveZone);

zoneCanvas().addEventListener("click", (event) => {
  if (!zoneDraw.active) return;
  const box = videoBox();
  if (!box) return;
  const rect = zoneCanvas().getBoundingClientRect();
  const cx = event.clientX - rect.left;
  const cy = event.clientY - rect.top;
  /* Ignore clicks on the letterbox black bars outside the video box. */
  if (cx < box.x || cx > box.x + box.w || cy < box.y || cy > box.y + box.h) return;
  zoneDraw.points.push([(cx - box.x) / box.w, (cy - box.y) / box.h]);
  el.saveZoneBtn.disabled = zoneDraw.points.length < 3;
  scheduleZonePreview();
});

/* Re-fit on resize so the preview keeps mapping onto the letterboxed video. */
window.addEventListener("resize", () => {
  if (zoneDraw.active) { fitZoneCanvas(); scheduleZonePreview(); }
});

/* MJPEG arrives as multipart/x-mixed-replace, which only an <img> renders.
   Assigning src opens the stream; removing it closes the connection. */
function attachStream(nodeId) {
  if (!nodeId) return;
  el.feed.src = API_BASE + "/video-feed/" + nodeId;
  setFocus(nodeId);
}

function clearStream() {
  el.feed.removeAttribute("src");
  setFocus("none");
}

function openCounting(id) {
  const node = nodes.get(id);
  if (!node) return;

  selectedId = id;
  renderAnalytics(node);

  /* Leaflet fires marker clicks after a double-click timeout, so a rapid tap
     can arrive after Escape already returned to the map. Ignore the stale one. */
  if (performance.now() - lastEscapeAt < 500) return;
  showView("count");
}

el.backBtn.addEventListener("click", () => showView("map"));

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  lastEscapeAt = performance.now();
  if (el.viewCount.classList.contains("is-visible")) showView("map");
});

/* ---- Analytics (VIEW 2) ----------------------------------------- */

function renderAnalytics(node) {
  const cls = stClass(node);

  el.controlStatus.textContent = "";
  el.panelPlace.textContent = node.nama;
  el.camChip.textContent = "CAM " +
    String([...nodes.keys()].indexOf(node.id) + 1).padStart(2, "0");
  el.camChip.className = "cam-chip " + cls;

  el.sumTotal.textContent = total(node);
  el.sumMotor.textContent = node.motor;
  el.sumMobil.textContent = node.mobil;
  el.sumTruk.textContent = node.truk;

  el.countMotor.textContent = node.motor;
  el.countMobil.textContent = node.mobil;
  el.countTruk.textContent = node.truk;

  el.avgHour.textContent = averagePerHour(node);
  el.peakHour.textContent = node.startTime ? node.peakOccupancy : 0;

  renderChart(node);
  renderGauge(node);
}

function averagePerHour(node) {
  if (!node.startTime) return 0;
  const elapsedHours = Math.max((Date.now() / 1000 - node.startTime) / 3600, 1);
  return Math.round(total(node) / elapsedHours);
}

function chartData(node) {
  const hours = Object.keys(node.hourlyCounts || {}).sort().slice(-6);
  return hours.map((hour) => ({
    hour: hour,
    value: Number(node.hourlyCounts[hour] || 0),
  }));
}

function renderChart(node) {
  const points = chartData(node);
  const peak = points.length ? Math.max(...points.map((point) => point.value)) : 0;

  el.chart.innerHTML = points.length ? points.map((point) => (
    '<div class="chart-bar' + (point.value === peak && peak > 0 ? " is-peak" : "") +
      '" title="' + point.hour + ':00: ' + point.value + ' kendaraan">' +
      '<i style="height:' +
        (peak > 0 ? Math.round((point.value / peak) * 100) : 0) + '%"></i>' +
    "</div>"
  )).join("") : '<span class="chart-empty">Belum ada data per jam.</span>';

  el.chartAxis.innerHTML = points.map((point) => (
    "<span>" + point.hour + "</span>"
  )).join("");

  el.chart.setAttribute(
    "aria-label",
    "Diagram batang jumlah kendaraan per jam. " +
    "Puncak " + peak + " kendaraan."
  );
}

function renderGauge(node) {
  const cls = stClass(node);
  const pct = Math.max(0, Math.min(100, node.occupancy));

  el.gauge.className = "gauge " + cls;
  el.gaugeFill.style.strokeDashoffset = GAUGE_CIRCUMFERENCE * (1 - pct / 100);
  el.occValue.textContent = pct + "%";

  el.gaugeStatus.className = "gauge-status " + cls;
  el.occLabel.textContent = rawStatus(node);
  el.occNote.textContent = NOTES[node.status] || "";
}

/* ---- Dynamic node registry -------------------------------------- */

/* /api/v1/nodes reads nodes.yaml fresh per request, so markers + sidebar
   cover every configured simpang the moment the page boots. */
async function loadRegistry() {
  const res = await fetch(API_BASE + "/nodes", { cache: "no-store" });
  if (!res.ok) throw new Error("HTTP " + res.status);
  const payload = await res.json();
  (payload.nodes || []).forEach((incoming) => {
    const prev = nodes.get(incoming.id);
    nodes.set(incoming.id, {
      id: incoming.id,
      nama: incoming.nama || (prev && prev.nama) || incoming.id,
      lat: incoming.lat != null ? incoming.lat : (prev && prev.lat),
      lng: incoming.lng != null ? incoming.lng : (prev && prev.lng),
      motor: (prev && prev.motor) || 0,
      mobil: (prev && prev.mobil) || 0,
      truk: (prev && prev.truk) || 0,
      occupancy: (prev && prev.occupancy) || 0,
      online: prev ? prev.online : false,
      status: (prev && prev.status) || "OFFLINE",
      startTime: (prev && prev.startTime) || 0,
      peakOccupancy: (prev && prev.peakOccupancy) || 0,
      hourlyCounts: (prev && prev.hourlyCounts) || {},
    });
  });
}

/* ---- Live polling ----------------------------------------------- */

/* Overlay tracker snapshots on the registry identity; unknown ids are added
   so a node that appears in /traffic-data without /nodes still shows up. */
function applyNodes(apiNodes) {
  Object.entries(apiNodes).forEach(([id, incoming]) => {
    const prev = nodes.get(id);
    nodes.set(id, {
      id: id,
      nama: incoming.nama || (prev && prev.nama) || id,
      lat: incoming.lat != null ? incoming.lat : (prev && prev.lat),
      lng: incoming.lng != null ? incoming.lng : (prev && prev.lng),
      motor: incoming.motor || 0,
      mobil: incoming.mobil || 0,
      truk: incoming.truk || 0,
      occupancy: incoming.occupancy || 0,
      online: incoming.online != null
        ? Boolean(incoming.online) : (prev && prev.online) || false,
      status: incoming.status || (prev && prev.status) || "OFFLINE",
      startTime: incoming.start_time != null
        ? Number(incoming.start_time) : (prev && prev.startTime) || 0,
      peakOccupancy: incoming.peak_occupancy != null
        ? Number(incoming.peak_occupancy) : (prev && prev.peakOccupancy) || 0,
      hourlyCounts: incoming.hourly_counts != null
        ? incoming.hourly_counts : (prev && prev.hourlyCounts) || {},
    });
  });

  upsertMarkers();
  renderMenu();
  if (selectedId && nodes.has(selectedId)) renderAnalytics(nodes.get(selectedId));
}

async function poll() {
  try {
    const res = await fetch(API_BASE + "/traffic-data", { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const payload = await res.json();
    applyNodes(payload.nodes || {});
  } catch (err) {
    /* Backend unreachable: keep whatever is on screen. */
  }
}

/* ---- Clock (format Indonesia) ----------------------------------- */

function tickClock() {
  const now = new Date();

  el.clockTime.textContent = now.toLocaleTimeString("id-ID", {
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  });
  el.clockTime.dateTime = now.toISOString();

  el.clockDate.textContent = now.toLocaleDateString("id-ID", {
    weekday: "long", day: "numeric", month: "long", year: "numeric",
  });
}

/* ---- Boot ------------------------------------------------------- */

/* Chrome note: if the Leaflet CDN is blocked, `L` is undefined and every map
   call throws. Letting that abort boot() is what froze the clock and left the
   page blank, so each step below is isolated: a dead map or a dead backend
   degrades that feature only, and the clock + poll always come up. */
async function boot() {
  tickClock();
  setInterval(tickClock, 1000);

  try {
    initMap();
  } catch (err) {
    console.warn("[ATCS] map init failed (Leaflet CDN blocked?):", err.message);
  }

  try {
    await loadRegistry();
  } catch (err) {
    console.warn("[ATCS] node registry unavailable, retrying via poll:", err.message);
  }

  try {
    upsertMarkers();
    renderMenu();
  } catch (err) {
    console.warn("[ATCS] initial render failed:", err.message);
  }

  poll();
  setInterval(poll, POLL_MS);
}

boot();
