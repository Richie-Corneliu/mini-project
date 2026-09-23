/* Smart ATCS Banyumas - Command Center
   MOCK_NODES renders instantly so the console never boots empty; the FastAPI
   backend then overlays live counts on a 1.5s poll. If the API is unreachable
   the mock simply stays on screen.

   Two views share one page:
     view-map    WebGIS overview, Leaflet map plus the floating dashboard menu
     view-count  full-bleed MJPEG stream with the analytics drawer along the bottom
*/

"use strict";

const MAP_CENTER = [-7.4245, 109.2302];
const MAP_ZOOM = 14;
const TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";

/* Backend: FastAPI serves processed MJPEG + tracker JSON. The frontend never
   touches RTSP/HLS directly, only these HTTP endpoints. */
const API_BASE = "http://localhost:8000/api/v1";
const POLL_MS = 1500;

const STATUS_ORDER = ["LANCAR", "PADAT", "MACET"];
const STATUS_CLASS = { LANCAR: "st-LANCAR", PADAT: "st-PADAT", MACET: "st-MACET" };

const NOTES = {
  LANCAR: "Arus lalu lintas lancar, tidak ada penumpukan kendaraan.",
  PADAT: "Volume kendaraan meningkat, pantau pergerakan tiap 5 menit.",
  MACET: "Kepadatan tinggi. Pengaturan simpang perlu dikoordinasikan.",
};

const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 50;

/* ---- Mock data -------------------------------------------------- *
   Three simpang held in Purwokerto. hourly[] holds 08:00 through 13:59. */

const MOCK_NODES = [
  {
    id: "node_01",
    nama: "Simpang Kebon Dalem",
    lat: -7.4245,
    lng: 109.2302,
    motor: 100,
    mobil: 40,
    truk: 10,
    occupancy: 48,
    status: "PADAT",
    hourly: [96, 128, 142, 118, 134, 104],
  },
  {
    id: "node_02",
    nama: "Simpang Tanjung",
    lat: -7.4302,
    lng: 109.2401,
    motor: 62,
    mobil: 22,
    truk: 4,
    occupancy: 17,
    status: "LANCAR",
    hourly: [88, 104, 96, 112, 90, 76],
  },
  {
    id: "node_03",
    nama: "Simpang Pasar Wage",
    lat: -7.4188,
    lng: 109.2389,
    motor: 156,
    mobil: 71,
    truk: 18,
    occupancy: 74,
    status: "MACET",
    hourly: [142, 186, 208, 176, 198, 164],
  },
];

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

  drawer: byId("drawer"),
  drawerHandle: byId("drawer-handle"),
  drawerToggle: byId("drawer-toggle"),
  toggleLabel: byId("toggle-label"),
  drawerPlace: byId("drawer-place"),
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
const statusOf = (node) => STATUS_CLASS[node.status] || "st-LANCAR";

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
    className: "veh-marker " + statusOf(node),
    html: '<div class="ring"></div><div class="core"></div><div class="pin-label"></div>',
    iconSize: [44, 44],
    iconAnchor: [22, 22],
  });
}

/* Marker creation and refresh share one path so a poll can recolor a pin in
   place. The icon is rebuilt only when the status actually changed, which
   keeps the pulse animation from restarting on every poll cycle. */
function upsertMarkers() {
  nodes.forEach((node) => {
    if (node.lat == null || node.lng == null) return;

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

      marker.addTo(map);
      marker._status = node.status;
      markers.set(node.id, marker);
      return;
    }

    existing.setLatLng([node.lat, node.lng]);
    if (existing._status !== node.status) {
      existing.setIcon(markerIcon(node));
      existing._status = node.status;
    }
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
    const count = list.filter((node) => node.status === status).length;
    const seg = el.split[status];
    seg.dataset.count = count;
    seg.style.flexGrow = count || 0.001;
    seg.title = status + ": " + count + " simpang";
  });

  el.menuList.innerHTML = list.map((node) => (
    '<li><button type="button" class="menu-item" data-id="' + node.id +
    '" data-status="' + statusOf(node) + '">' +
      '<span class="menu-dot"></span>' +
      '<span class="menu-name">' + node.nama + "</span>" +
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

  if (toCounting) settle(true);
}

/* ---- Backend bridge (focus + MJPEG) ----------------------------- */

/* Ask the worker to prioritise this node's inference. Fire-and-forget: the
   endpoint is optional, so a missing backend must not break the UI. */
function setFocus(nodeId) {
  fetch(API_BASE + "/set-focus/" + nodeId, { method: "POST" }).catch(() => {});
}

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

/* ---- Analytics (VIEW 2) ----------------------------------------- */

function renderAnalytics(node) {
  const grand = total(node);
  const cls = statusOf(node);
  const hourly = node.hourly || [];

  el.drawerPlace.textContent = node.nama;
  el.camChip.textContent = "CAM " + String([...nodes.keys()].indexOf(node.id) + 1).padStart(2, "0");
  el.camChip.className = "cam-chip " + cls;

  el.sumTotal.textContent = grand;
  el.sumMotor.textContent = node.motor;
  el.sumMobil.textContent = node.mobil;
  el.sumTruk.textContent = node.truk;

  el.countMotor.textContent = node.motor;
  el.countMobil.textContent = node.mobil;
  el.countTruk.textContent = node.truk;

  el.avgHour.textContent = hourly.length
    ? Math.round(hourly.reduce((a, b) => a + b, 0) / hourly.length)
    : 0;
  el.peakHour.textContent = hourly.length ? Math.max(...hourly) : 0;

  renderChart(node);
  renderGauge(node);
}

/* The poll only refreshes what the backend actually tracks, so the hourly
   chart is left alone instead of being rebuilt every 1.5 seconds. */
function renderLiveMetrics(node) {
  el.sumTotal.textContent = total(node);
  el.sumMotor.textContent = node.motor;
  el.sumMobil.textContent = node.mobil;
  el.sumTruk.textContent = node.truk;

  el.countMotor.textContent = node.motor;
  el.countMobil.textContent = node.mobil;
  el.countTruk.textContent = node.truk;

  el.camChip.className = "cam-chip " + statusOf(node);
  renderGauge(node);
}

function renderChart(node) {
  const hourly = node.hourly || [];
  const peak = hourly.length ? Math.max(...hourly) : 0;

  el.chart.innerHTML = hourly.map((value) => (
    '<div class="chart-bar' + (value === peak ? " is-peak" : "") + '">' +
      '<i style="height:' + (peak > 0 ? Math.round((value / peak) * 100) : 0) + '%"></i>' +
    "</div>"
  )).join("");

  el.chart.setAttribute(
    "aria-label",
    "Diagram batang jumlah kendaraan per jam, 08:00 sampai 13:59. " +
    "Puncak " + peak + " kendaraan."
  );
}

function renderGauge(node) {
  const pct = Math.max(0, Math.min(100, node.occupancy));

  el.gauge.className = "gauge " + statusOf(node);
  el.gaugeFill.style.strokeDashoffset = GAUGE_CIRCUMFERENCE * (1 - pct / 100);
  el.occValue.textContent = pct + "%";

  el.gaugeStatus.className = "gauge-status " + statusOf(node);
  el.occLabel.textContent = node.status;
  el.occNote.textContent = NOTES[node.status] || "";
}

/* ---- Drawer: expand, collapse, drag ----------------------------- */

const drawer = el.drawer;
let drawerOpen = false;

/* How far the panel sits below its expanded position while collapsed.
   --drawer-peek is a plain px value so it reads back fine, but
   --drawer-panel-h is a clamp(), and a custom property hands back the raw
   token stream rather than a resolved length, so the rest is measured. */
function panelHeight() {
  const peek = parseFloat(getComputedStyle(drawer).getPropertyValue("--drawer-peek"));
  return Math.max(0, drawer.offsetHeight - (Number.isFinite(peek) ? peek : 0));
}

function dragTo(offset) {
  drawer.classList.add("is-dragging");
  drawer.style.transform = "translateX(-50%) translateY(" + offset + "px)";
}

/* Handing control back to the class means dropping the inline transform,
   which is safe because the class transform and px 0 describe the same place. */
function settle(open) {
  drawerOpen = open;
  drawer.classList.remove("is-dragging");
  drawer.style.transform = "";
  drawer.classList.toggle("is-open", open);
  el.drawerToggle.setAttribute("aria-expanded", String(open));
  el.drawerHandle.setAttribute("aria-expanded", String(open));
  el.toggleLabel.textContent = open ? "Tutup" : "Rincian";
}

el.drawerToggle.addEventListener("click", () => settle(!drawerOpen));
/* ---- Drag to expand / collapse ---------------------------------- *
   Tracking runs on document rather than the handle. A capture on the handle
   alone is not enough: if the pointer leaves it mid-gesture the moves stop
   arriving, and the panel freezes halfway. */

const drag = { active: false, startY: 0, startOffset: 0, moved: false, pointerId: null };

function clampedOffset(clientY) {
  const span = panelHeight();
  const travel = drag.startOffset + (clientY - drag.startY);
  return Math.max(0, Math.min(span, travel));
}

el.drawerHandle.addEventListener("pointerdown", (event) => {
  drag.active = true;
  drag.moved = false;
  drag.pointerId = event.pointerId;
  drag.startY = event.clientY;
  drag.startOffset = drawerOpen ? 0 : panelHeight();
  drawer.classList.add("is-dragging");
  event.preventDefault();
});

document.addEventListener("pointermove", (event) => {
  if (!drag.active || event.pointerId !== drag.pointerId) return;
  if (Math.abs(event.clientY - drag.startY) > 4) drag.moved = true;
  if (drag.moved) dragTo(clampedOffset(event.clientY));
});

function endDrag(event) {
  if (!drag.active || event.pointerId !== drag.pointerId) return;
  drag.active = false;

  /* A press that never moved is a tap, and a tap on the handle toggles. */
  if (!drag.moved) {
    settle(!drawerOpen);
    return;
  }

  settle(clampedOffset(event.clientY) < panelHeight() / 2);
}

document.addEventListener("pointerup", endDrag);
document.addEventListener("pointercancel", endDrag);

/* The handle is a real button, so Enter and Space already toggle it. */
el.drawerHandle.addEventListener("keydown", (event) => {
  if (event.key === "ArrowUp" && !drawerOpen) { event.preventDefault(); settle(true); }
  if (event.key === "ArrowDown" && drawerOpen) { event.preventDefault(); settle(false); }
});

/* Collapse when the pointer goes down anywhere outside an open drawer. */
document.addEventListener("pointerdown", (event) => {
  if (!drawerOpen) return;
  if (drawer.contains(event.target)) return;
  if (el.backBtn.contains(event.target)) return;
  settle(false);
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  lastEscapeAt = performance.now();
  /* Only the frontmost layer steps back, so the drawer closes before the view. */
  if (drawerOpen) settle(false);
  else if (el.viewCount.classList.contains("is-visible")) showView("map");
});

/* A resize changes the vh-derived panel height, so the collapsed offset has
   to be re-applied or the panel drifts off its resting position. */
window.addEventListener("resize", () => {
  if (drag.active) return;
  drawer.classList.add("is-dragging");
  drawer.style.transform = drawerOpen ? "translateX(-50%) translateY(0px)"
    : "translateX(-50%) translateY(" + panelHeight() + "px)";
  void drawer.offsetWidth;
  drawer.classList.remove("is-dragging");
});

/* ---- Live polling ----------------------------------------------- */

/* Merge tracker snapshots over the mock baseline. The API carries counts and
   status but no hourly breakdown, so the chart series is inherited from the
   mock entry rather than nulled out. */
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
      status: incoming.status || "LANCAR",
      hourly: prev ? prev.hourly : null,
    });
  });

  renderMenu();
  if (selectedId && nodes.has(selectedId)) renderLiveMetrics(nodes.get(selectedId));
}

async function poll() {
  try {
    const res = await fetch(API_BASE + "/traffic-data", { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const payload = await res.json();
    applyNodes(payload.nodes || {});
  } catch (err) {
    /* Backend unreachable: the mock baseline is already on screen, so the
       console just keeps showing it instead of blanking out. */
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

MOCK_NODES.forEach((node) => nodes.set(node.id, node));
initMap();
upsertMarkers();
renderMenu();
tickClock();
setInterval(tickClock, 1000);
renderAnalytics(nodes.get(MOCK_NODES[0].id));
settle(false);

poll();
setInterval(poll, POLL_MS);
