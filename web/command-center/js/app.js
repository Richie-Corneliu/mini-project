/* Smart ATCS Banyumas - Command Center
   Node identity (markers + sidebar) comes from /api/v1/nodes, which reads
   nodes.yaml fresh per request, so a new simpang appears without a redeploy.
   /api/v1/traffic-data then overlays live counts on a 1.5s poll; markers,
   preview card, and sidebar repaint from the same snapshot.

   Two views share one page:
     view-map    WebGIS overview, OpenStreetMap dark basemap + floating frosted drawer
     view-count  fixed-height split screen: 3-row stream viewport (left),
                 460px analytics console (right)
*/

"use strict";

const MAP_CENTER = [-7.4245, 109.2302];
const MAP_ZOOM = 14;

/* Free, keyless OpenStreetMap default (dark-styled via CSS tile-pane filter).
   Stadia and MapTiler remain available via ?tiles=stadia or ?tiles=maptiler. */
const FREE_KEYS = {
  maptiler: "",
  stadia: "",
};

const TILE_PROVIDERS = {
  osm: {
    url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    subdomains: "",
    maxZoom: 19,
    attr: "&copy; OpenStreetMap contributors",
  },
  stadia: {
    url: "https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png",
    subdomains: "",
    maxZoom: 20,
    attr: '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  },
  maptiler: {
    url: "https://api.maptiler.com/maps/dataviz-dark/{z}/{x}/{y}@2x.png?key={key}",
    subdomains: "",
    maxZoom: 20,
    attr: '&copy; <a href="https://www.maptiler.com/copyright/">MapTiler</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  },
};

function pickTileProvider() {
  const want = new URLSearchParams(location.search).get("tiles");
  const chosen = TILE_PROVIDERS[want];
  if (!chosen) return TILE_PROVIDERS.osm;
  if (chosen.url.includes("{key}")) {
    const key = FREE_KEYS[want];
    if (!key) {
      console.warn("[ATCS] no key for tiles=" + want + ", using OpenStreetMap");
      return TILE_PROVIDERS.osm;
    }
    return Object.assign({}, chosen, { url: chosen.url.replace("{key}", key) });
  }
  return chosen;
}

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

/* ---- DOM refs --------------------------------------------------- */

const byId = (id) => document.getElementById(id);

const el = {
  viewMap: byId("view-map"),
  viewCount: byId("view-count"),
  feed: byId("stage-video"),
  backBtn: byId("back-btn"),
  camChip: byId("cam-chip"),
  viewportLive: byId("viewport-live"),
  viewportLiveText: byId("viewport-live-text"),
  liveBadge: byId("live-badge"),
  liveBadgeText: byId("live-badge-text"),
  streamOverlay: byId("stream-overlay"),

  clockTime: byId("clock-time"),
  clockDate: byId("clock-date"),

  mapFilterBar: byId("map-filter-bar"),
  filterCountAll: byId("filter-count-all"),
  filterCountLancar: byId("filter-count-lancar"),
  filterCountCongested: byId("filter-count-congested"),

  previewCard: byId("map-preview-card"),
  previewName: byId("preview-name"),
  previewStatus: byId("preview-status"),
  previewCloseBtn: byId("preview-close-btn"),
  previewOccVal: byId("preview-occ-val"),
  previewOccFill: byId("preview-occ-fill"),
  previewMotor: byId("preview-motor"),
  previewMobil: byId("preview-mobil"),
  previewTruk: byId("preview-truk"),
  previewOpenBtn: byId("preview-open-btn"),

  menuCount: byId("menu-count"),
  menuList: byId("menu-list"),
  totalKendaraan: byId("total-kendaraan"),
  totalSimpang: byId("total-simpang"),
  classSplit: {
    motor: byId("class-split-motor"),
    mobil: byId("class-split-mobil"),
    truk: byId("class-split-truk"),
  },
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
  exportCsvBtn: byId("btn-export-csv"),
  sumTotal: byId("sum-total"),
  sumMotor: byId("sum-motor"),
  sumMobil: byId("sum-mobil"),
  sumTruk: byId("sum-truk"),

  countMotor: byId("count-motor"),
  countMobil: byId("count-mobil"),
  countTruk: byId("count-truk"),
  avgHour: byId("avg-hour"),
  avgBasis: byId("avg-basis"),
  peakHour: byId("peak-hour"),

  datePickerBtn: byId("btn-date-picker"),
  datePickerLabel: byId("date-picker-label"),
  calPopover: byId("cal-popover"),
  calPrev: byId("cal-prev"),
  calNext: byId("cal-next"),
  calTodayBtn: byId("cal-today-btn"),
  calMonthLabel: byId("cal-month-label"),
  calGrid: byId("cal-grid"),

  chartSub: byId("chart-sub"),
  chart: byId("chart"),
  chartAxis: byId("chart-axis"),
  slotHourSelect: byId("slot-hour-select"),
  slotDetailBody: byId("slot-detail-body"),
  slotDetailFoot: byId("slot-detail-foot"),
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
let previewNodeId = null;
let previewPinned = false;
let mapFilter = "ALL";
let lastEscapeAt = 0;

let todayWibDate = new Date().toLocaleDateString("sv-SE", { timeZone: "Asia/Jakarta" });
let selectedDate = null;
let selectedHour = null;
let historyDates = new Set();
const historicalNodes = new Map();
let calViewYear = Number(todayWibDate.slice(0, 4)) || new Date().getFullYear();
let calViewMonth = (Number(todayWibDate.slice(5, 7)) || 1) - 1;
let lastDatesFetchAt = 0;

const total = (node) => node.motor + node.mobil + node.truk;
const fmtNum = (n) => Number(n || 0).toLocaleString("id-ID");

const STREAM_BADGE = {
  LIVE: { text: "COMMAND CENTER - LIVE", cls: "" },
  RECONNECTING: { text: "STREAM RECONNECTING...", cls: "status-reconnecting" },
  OFFLINE: { text: "STREAM OFFLINE", cls: "status-offline" },
};

function streamStatusOf(node) {
  if (!node) return "OFFLINE";
  if (node.streamStatus) return node.streamStatus;
  return node.online ? "LIVE" : "OFFLINE";
}

function activeStreamStatus() {
  if (selectedId && nodes.has(selectedId)) {
    return streamStatusOf(nodes.get(selectedId));
  }
  const list = [...nodes.values()];
  if (list.some((n) => streamStatusOf(n) === "RECONNECTING")) return "RECONNECTING";
  if (list.some((n) => streamStatusOf(n) === "LIVE")) return "LIVE";
  return "OFFLINE";
}

function renderStreamHealth() {
  const status = activeStreamStatus();
  const meta = STREAM_BADGE[status] || STREAM_BADGE.OFFLINE;
  el.liveBadge.className = "live-badge" + (meta.cls ? " " + meta.cls : "");
  el.liveBadgeText.textContent = meta.text;
  if (el.viewportLive) {
    el.viewportLive.className = "viewport-live" + (meta.cls ? " " + meta.cls : "");
    if (el.viewportLiveText) {
      el.viewportLiveText.textContent =
        status === "RECONNECTING" ? "RECONNECT" : status;
    }
  }
  el.streamOverlay.hidden = status !== "RECONNECTING";
}

function rawStatus(node) {
  if (!node.online) return "OFFLINE";
  return STATUS_ORDER.includes(node.status) ? node.status : "unknown";
}
function stClass(node) {
  return STATUS_CLASS[rawStatus(node)] || "st-unknown";
}

function matchesMapFilter(node) {
  if (mapFilter === "ALL") return true;
  const st = rawStatus(node);
  if (mapFilter === "LANCAR") return st === "LANCAR";
  if (mapFilter === "CONGESTED") return st === "PADAT" || st === "MACET";
  return true;
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;",
    '"': "&quot;", "'": "&#39;",
  })[c]);
}

/* ---- Map & Floating Preview ------------------------------------- */

function initMap() {
  const prov = pickTileProvider();
  map = L.map("map", { center: MAP_CENTER, zoom: MAP_ZOOM });
  const tileOpts = {
    maxZoom: prov.maxZoom,
    detectRetina: true,
    attribution: prov.attr,
  };
  if (prov.subdomains) tileOpts.subdomains = prov.subdomains;
  L.tileLayer(prov.url, tileOpts).addTo(map);
  map.on("click", () => closePreviewCard());
}

function markerIcon(node, isSelected, isFilteredOut) {
  const cls = [
    "veh-marker",
    stClass(node),
    isSelected ? "is-selected" : "",
    isFilteredOut ? "is-filtered-out" : "",
  ].filter(Boolean).join(" ");
  return L.divIcon({
    className: cls,
    html:
      '<div class="ping"></div><div class="ping ping-2"></div>' +
      '<div class="core"></div>' +
      '<div class="pin-label">' + escapeHtml(node.nama) + "</div>",
    iconSize: [44, 44],
    iconAnchor: [22, 22],
  });
}

function renderPreviewCard() {
  if (!el.previewCard) return;
  const node = previewNodeId ? nodes.get(previewNodeId) : null;
  if (!node) {
    el.previewCard.hidden = true;
    return;
  }
  const st = rawStatus(node);
  const cls = stClass(node);
  const occ = Math.max(0, Math.min(100, Number(node.occupancy) || 0));

  el.previewName.textContent = node.nama;
  el.previewStatus.textContent = st;
  el.previewStatus.className = "status-pill " + cls;
  el.previewOccVal.textContent = occ + "%";
  el.previewOccFill.style.width = occ + "%";
  el.previewOccFill.className = "preview-occ-fill " + cls;
  el.previewMotor.textContent = fmtNum(node.motor);
  el.previewMobil.textContent = fmtNum(node.mobil);
  el.previewTruk.textContent = fmtNum(node.truk);
  el.previewCard.hidden = false;
}

function openPreviewCard(id, pin) {
  if (!nodes.has(id)) return;
  previewNodeId = id;
  if (pin) previewPinned = true;
  renderPreviewCard();
  upsertMarkers();
  renderMenu();
}

function closePreviewCard() {
  previewNodeId = null;
  previewPinned = false;
  renderPreviewCard();
  upsertMarkers();
  renderMenu();
}

if (el.previewCloseBtn) {
  el.previewCloseBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    closePreviewCard();
  });
}

if (el.previewOpenBtn) {
  el.previewOpenBtn.addEventListener("click", () => {
    if (previewNodeId) openCounting(previewNodeId);
  });
}

function upsertMarkers() {
  if (!map) return;

  nodes.forEach((node) => {
    if (node.lat == null || node.lng == null) return;

    const status = stClass(node);
    const isSel = node.id === previewNodeId;
    const isOut = !matchesMapFilter(node);
    const iconKey = status + "|" + (isSel ? "1" : "0") + "|" + (isOut ? "1" : "0");
    const existing = markers.get(node.id);

    if (!existing) {
      const marker = L.marker([node.lat, node.lng], {
        icon: markerIcon(node, isSel, isOut),
        riseOnHover: true,
        keyboard: true,
        title: node.nama,
        alt: node.nama,
      });

      marker.on("mouseover", () => {
        if (!previewPinned) openPreviewCard(node.id, false);
      });
      marker.on("click", (ev) => {
        L.DomEvent.stopPropagation(ev);
        if (previewNodeId === node.id && previewPinned) {
          openCounting(node.id);
        } else {
          openPreviewCard(node.id, true);
        }
      });
      marker.on("dblclick", () => openCounting(node.id));

      marker.addTo(map);
      marker._iconKey = iconKey;
      markers.set(node.id, marker);
      return;
    }

    existing.setLatLng([node.lat, node.lng]);
    if (existing._iconKey !== iconKey) {
      existing.setIcon(markerIcon(node, isSel, isOut));
      existing._iconKey = iconKey;
    }
  });
}

/* ---- Map Filter Bar & Dashboard Drawer (VIEW 1) ----------------- */

if (el.mapFilterBar) {
  el.mapFilterBar.addEventListener("click", (event) => {
    const btn = event.target.closest(".filter-chip");
    if (!btn || !btn.dataset.filter) return;
    mapFilter = btn.dataset.filter;
    el.mapFilterBar.querySelectorAll(".filter-chip").forEach((chip) => {
      const active = chip.dataset.filter === mapFilter;
      chip.classList.toggle("is-active", active);
      chip.setAttribute("aria-pressed", active ? "true" : "false");
    });
    upsertMarkers();
    renderMenu();
  });
}

function renderMenu() {
  const list = [...nodes.values()];
  const grand = list.reduce((sum, node) => sum + total(node), 0);
  const sumMotor = list.reduce((sum, node) => sum + (Number(node.motor) || 0), 0);
  const sumMobil = list.reduce((sum, node) => sum + (Number(node.mobil) || 0), 0);
  const sumTruk = list.reduce((sum, node) => sum + (Number(node.truk) || 0), 0);

  el.totalKendaraan.textContent = fmtNum(grand);
  el.totalSimpang.textContent = list.length;
  el.menuCount.textContent = list.length + " titik";

  const lancarCnt = list.filter((n) => rawStatus(n) === "LANCAR").length;
  const congestedCnt = list.filter((n) => {
    const st = rawStatus(n);
    return st === "PADAT" || st === "MACET";
  }).length;
  if (el.filterCountAll) el.filterCountAll.textContent = list.length;
  if (el.filterCountLancar) el.filterCountLancar.textContent = lancarCnt;
  if (el.filterCountCongested) el.filterCountCongested.textContent = congestedCnt;

  const classMap = { motor: sumMotor, mobil: sumMobil, truk: sumTruk };
  ["motor", "mobil", "truk"].forEach((k) => {
    const seg = el.classSplit[k];
    if (!seg) return;
    seg.dataset.count = classMap[k];
    seg.style.flexGrow = classMap[k] || 0.001;
  });

  STATUS_ORDER.forEach((status) => {
    const count = list.filter((node) => rawStatus(node) === status).length;
    const seg = el.split[status];
    if (!seg) return;
    seg.dataset.count = count;
    seg.style.flexGrow = count || 0.001;
    seg.title = status + ": " + count + " simpang";
  });

  const visible = list.filter(matchesMapFilter);
  if (!visible.length) {
    el.menuList.innerHTML = '<li class="menu-empty">Tidak ada simpang pada filter ini.</li>';
    return;
  }

  el.menuList.innerHTML = visible.map((node) => {
    const st = rawStatus(node);
    const cls = stClass(node);
    const vol = total(node);
    const occ = Math.max(0, Math.min(100, Number(node.occupancy) || 0));
    const m = Number(node.motor) || 0;
    const c = Number(node.mobil) || 0;
    const t = Number(node.truk) || 0;
    const denom = vol || 1;
    const pm = vol ? (m / denom) * 100 : 33.4;
    const pc = vol ? (c / denom) * 100 : 33.3;
    const pt = vol ? (t / denom) * 100 : 33.3;
    const selCls = node.id === previewNodeId ? " is-selected" : "";

    return (
      '<li><button type="button" class="menu-item' + selCls + '" data-id="' + node.id +
      '" data-status="' + st + '">' +
        '<div class="menu-card-top">' +
          '<span class="menu-title-group">' +
            '<span class="menu-dot"></span>' +
            '<span class="menu-name">' + escapeHtml(node.nama) + "</span>" +
          "</span>" +
          '<span class="status-pill ' + cls + '">' + st + "</span>" +
        "</div>" +
        '<div class="menu-card-mid">' +
          '<span class="menu-vol-wrap">' +
            '<span class="menu-val">' + fmtNum(vol) + "</span>" +
            '<span class="menu-unit">kendaraan</span>' +
          "</span>" +
          '<span class="menu-occ">Okupansi ' + occ + "%</span>" +
        "</div>" +
        '<div class="menu-class-bar" aria-hidden="true">' +
          '<i class="cb-motor" style="width:' + pm.toFixed(1) + '%"></i>' +
          '<i class="cb-mobil" style="width:' + pc.toFixed(1) + '%"></i>' +
          '<i class="cb-truk" style="width:' + pt.toFixed(1) + '%"></i>' +
        "</div>" +
        '<div class="menu-breakdown">' +
          '<span><i class="bd-motor"></i>' + fmtNum(m) + " Mtr</span>" +
          '<span><i class="bd-mobil"></i>' + fmtNum(c) + " Mbl</span>" +
          '<span><i class="bd-truk"></i>' + fmtNum(t) + " Trk</span>" +
        "</div>" +
      "</button></li>"
    );
  }).join("");
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

  if (!toCounting) {
    cancelZoneDraw();
    selectedId = null;
    clearStream();
    renderStreamHealth();
    if (map) map.invalidateSize();
    return;
  }

  attachStream(selectedId);
}

/* ---- Backend bridge (focus + MJPEG) ----------------------------- */

function getApiKey() {
  try {
    const qp = new URLSearchParams(window.location.search).get("api_key");
    if (qp && qp.trim()) {
      sessionStorage.setItem("atcs_api_key", qp.trim());
      return qp.trim();
    }
    return (sessionStorage.getItem("atcs_api_key") || "").trim();
  } catch (err) {
    return "";
  }
}

function postHeaders(extra) {
  const headers = Object.assign({}, extra || {});
  const key = getApiKey();
  if (key) headers["X-API-Key"] = key;
  return headers;
}

async function postControlFetch(url, options, allowPrompt) {
  const opts = Object.assign({ method: "POST" }, options || {});
  opts.headers = postHeaders(options && options.headers);
  let res = await fetch(url, opts);
  if (res.status === 401 && allowPrompt && typeof window.prompt === "function") {
    const entered = window.prompt("Masukkan API Key (ATCS_API_KEY) untuk kontrol:");
    if (entered && entered.trim()) {
      try { sessionStorage.setItem("atcs_api_key", entered.trim()); } catch (err) { }
      opts.headers = postHeaders(options && options.headers);
      res = await fetch(url, opts);
    }
  }
  return res;
}

function setFocus(nodeId) {
  const allowPrompt = nodeId !== "none";
  postControlFetch(API_BASE + "/set-focus/" + encodeURIComponent(nodeId), null, allowPrompt)
    .then((res) => {
      if (res.status === 401) {
        el.controlStatus.textContent = "Akses stream ditolak (ATCS_API_KEY tidak valid).";
      }
    })
    .catch(() => { });
}

async function sendControl(action) {
  if (!selectedId) return;

  el.resetCounterBtn.disabled = true;
  el.toggleRecordBtn.disabled = true;
  el.controlStatus.textContent = "Mengirim perintah...";
  try {
    const res = await postControlFetch(
      API_BASE + "/control/" + encodeURIComponent(selectedId) + "/" + action,
      null,
      true
    );
    if (res.status === 401) throw new Error("API Key diperlukan / tidak valid (401)");
    if (!res.ok) throw new Error("HTTP " + res.status);
    el.controlStatus.textContent = action === "reset"
      ? "Perintah reset diterima."
      : "Perintah rekam diterima.";
  } catch (err) {
    el.controlStatus.textContent = "Perintah gagal dikirim: " + err.message;
  } finally {
    el.resetCounterBtn.disabled = false;
    el.toggleRecordBtn.disabled = false;
  }
}

el.resetCounterBtn.addEventListener("click", () => sendControl("reset"));
el.toggleRecordBtn.addEventListener("click", () => sendControl("record"));

/* ---- Zone drawing (VIEW 2) --------------------------------------- */

const ZONE_COLORS = { line: "#38BDF8", fill: "rgba(56, 189, 248, 0.18)" };

const zoneDraw = {
  active: false,
  points: [],
  raf: null,
};

function zoneCanvas() {
  return document.getElementById("zone-canvas");
}

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
  ctx.lineTo(pts[0][0], pts[0][1]);
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
    const res = await postControlFetch(
      API_BASE + "/control/" + encodeURIComponent(selectedId) + "/update_zone",
      {
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ points: zoneDraw.points.map((p) => [p[0], p[1]]) }),
      },
      true
    );
    if (res.status === 401) throw new Error("API Key diperlukan / tidak valid (401)");
    if (!res.ok) throw new Error("HTTP " + res.status);
    cancelZoneDraw();
    await loadRegistry().catch(() => { });
    await poll();
    el.controlStatus.textContent = "Zona baru disimpan & aktif.";
  } catch (err) {
    el.saveZoneBtn.disabled = false;
    el.controlStatus.textContent = "Zona gagal dikirim: " + err.message;
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
  if (cx < box.x || cx > box.x + box.w || cy < box.y || cy > box.y + box.h) return;
  zoneDraw.points.push([(cx - box.x) / box.w, (cy - box.y) / box.h]);
  el.saveZoneBtn.disabled = zoneDraw.points.length < 3;
  scheduleZonePreview();
});

window.addEventListener("resize", () => {
  if (zoneDraw.active) { fitZoneCanvas(); scheduleZonePreview(); }
});

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
  selectedDate = null;
  selectedHour = null;
  historicalNodes.clear();
  closeCalendar();
  updateDatePickerButton();
  el.controlStatus.textContent = "";
  renderAnalytics(node);
  loadHistoryDates();

  if (performance.now() - lastEscapeAt < 500) return;
  showView("count");
}

el.backBtn.addEventListener("click", () => {
  closeCalendar();
  selectedDate = null;
  showView("map");
});

/* ---- Export CSV laporan harian simpang aktif --------------------- */

el.exportCsvBtn.addEventListener("click", async () => {
  if (!selectedId || !nodes.has(selectedId)) {
    el.controlStatus.textContent = "Pilih simpang terlebih dahulu.";
    return;
  }
  const btn = el.exportCsvBtn;
  const targetDate = selectedDate || todayWibDate;
  btn.disabled = true;
  btn.querySelector("span").textContent = "Mengunduh...";
  try {
    const query = selectedDate ? "?date=" + encodeURIComponent(selectedDate) : "";
    const res = await fetch(
      API_BASE + "/export-csv/" + encodeURIComponent(selectedId) + query
    );
    if (!res.ok) throw new Error("HTTP " + res.status);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "laporan_lalin_" + selectedId + "_" + targetDate + ".csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    el.controlStatus.textContent = "Laporan CSV berhasil diunduh.";
  } catch (err) {
    el.controlStatus.textContent = "Export CSV gagal: " + err.message;
  } finally {
    btn.disabled = false;
    btn.querySelector("span").textContent = "Export CSV";
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (!el.calPopover.hidden) {
    closeCalendar();
    return;
  }
  if (el.viewCount.classList.contains("is-visible")) {
    lastEscapeAt = performance.now();
    showView("map");
    return;
  }
  if (previewNodeId) closePreviewCard();
});

/* ---- Analytics & Historical Date View (VIEW 2) ------------------ */

const MONTH_NAMES_ID = [
  "Januari", "Februari", "Maret", "April", "Mei", "Juni",
  "Juli", "Agustus", "September", "Oktober", "November", "Desember",
];
const MONTH_SHORT_ID = [
  "Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
  "Jul", "Agu", "Sep", "Okt", "Nov", "Des",
];

function formatShortDateId(dateStr) {
  if (!dateStr || dateStr.length !== 10) return dateStr || "";
  const y = dateStr.slice(0, 4);
  const m = Number(dateStr.slice(5, 7)) - 1;
  const d = Number(dateStr.slice(8, 10));
  return d + " " + (MONTH_SHORT_ID[m] || "") + " " + y;
}

function updateDatePickerButton() {
  el.datePickerLabel.textContent = selectedDate
    ? formatShortDateId(selectedDate)
    : "Hari Ini";
  el.datePickerBtn.classList.toggle("is-historical", Boolean(selectedDate));
}

function closeCalendar() {
  el.calPopover.hidden = true;
  el.datePickerBtn.setAttribute("aria-expanded", "false");
}

function openCalendar() {
  const ref = selectedDate || todayWibDate;
  calViewYear = Number(ref.slice(0, 4)) || calViewYear;
  calViewMonth = (Number(ref.slice(5, 7)) || 1) - 1;
  renderCalendar();
  el.calPopover.hidden = false;
  el.datePickerBtn.setAttribute("aria-expanded", "true");
}

function renderCalendar() {
  el.calMonthLabel.textContent = MONTH_NAMES_ID[calViewMonth] + " " + calViewYear;
  const firstDow = new Date(Date.UTC(calViewYear, calViewMonth, 1)).getUTCDay();
  const daysInMonth = new Date(Date.UTC(calViewYear, calViewMonth + 1, 0)).getUTCDate();
  const activeDate = selectedDate || todayWibDate;
  const ym = String(calViewYear) + "-" + String(calViewMonth + 1).padStart(2, "0");

  const cells = [];
  for (let i = 0; i < firstDow; i++) {
    cells.push('<span class="cal-pad" aria-hidden="true"></span>');
  }
  for (let d = 1; d <= daysInMonth; d++) {
    const ds = ym + "-" + String(d).padStart(2, "0");
    const hasData = historyDates.has(ds);
    const isSelected = ds === activeDate;
    const isToday = ds === todayWibDate;
    const isFuture = todayWibDate && ds > todayWibDate;
    const cls = ["cal-day"];
    if (hasData) cls.push("has-data");
    if (isToday) cls.push("is-today");
    if (isSelected) cls.push("is-selected");
    if (isFuture) cls.push("is-future");
    cells.push(
      '<button type="button" class="' + cls.join(" ") + '" data-date="' + ds + '"' +
      (isFuture ? " disabled" : "") + ' aria-pressed="' + (isSelected ? "true" : "false") +
      '">' + d + "</button>"
    );
  }
  el.calGrid.innerHTML = cells.join("");
}

async function selectDate(dateStr) {
  const next = !dateStr || dateStr === todayWibDate ? null : dateStr;
  selectedDate = next;
  selectedHour = null;
  updateDatePickerButton();
  closeCalendar();

  if (!selectedDate) {
    historicalNodes.clear();
    if (selectedId && nodes.has(selectedId)) renderAnalytics(nodes.get(selectedId));
    return;
  }

  try {
    const res = await fetch(
      API_BASE + "/traffic-data?date=" + encodeURIComponent(selectedDate),
      { cache: "no-store" }
    );
    if (!res.ok) throw new Error("HTTP " + res.status);
    const payload = await res.json();
    historicalNodes.clear();
    Object.entries(payload.nodes || {}).forEach(([id, raw]) => {
      historicalNodes.set(id, mapNodePayload(id, raw, nodes.get(id)));
    });
    if (selectedId) {
      const target = historicalNodes.get(selectedId) || nodes.get(selectedId);
      if (target) renderAnalytics(target);
    }
  } catch (err) {
    el.controlStatus.textContent = "Gagal memuat data tanggal " + selectedDate + ".";
  }
}

async function loadHistoryDates() {
  lastDatesFetchAt = Date.now();
  try {
    const res = await fetch(API_BASE + "/history/dates", { cache: "no-store" });
    if (!res.ok) return;
    const payload = await res.json();
    if (payload.today) todayWibDate = payload.today;
    historyDates = new Set(payload.dates || []);
    if (!el.calPopover.hidden) renderCalendar();
  } catch (err) {
    /* Non-fatal: calendar still navigates dates manually. */
  }
}

el.datePickerBtn.addEventListener("click", (event) => {
  event.stopPropagation();
  el.calPopover.hidden ? openCalendar() : closeCalendar();
});

el.calPopover.addEventListener("click", (event) => event.stopPropagation());

el.calPrev.addEventListener("click", () => {
  calViewMonth -= 1;
  if (calViewMonth < 0) { calViewMonth = 11; calViewYear -= 1; }
  renderCalendar();
});

el.calNext.addEventListener("click", () => {
  calViewMonth += 1;
  if (calViewMonth > 11) { calViewMonth = 0; calViewYear += 1; }
  renderCalendar();
});

el.calTodayBtn.addEventListener("click", () => {
  calViewYear = Number(todayWibDate.slice(0, 4)) || calViewYear;
  calViewMonth = (Number(todayWibDate.slice(5, 7)) || 1) - 1;
  selectDate(todayWibDate);
});

el.calGrid.addEventListener("click", (event) => {
  const btn = event.target.closest(".cal-day");
  if (!btn || btn.disabled) return;
  selectDate(btn.dataset.date);
});

document.addEventListener("click", (event) => {
  if (el.calPopover.hidden) return;
  if (!el.calPopover.contains(event.target) && !el.datePickerBtn.contains(event.target)) {
    closeCalendar();
  }
});

function currentAnalyticsNode() {
  if (!selectedId) return null;
  if (selectedDate && historicalNodes.has(selectedId)) {
    return historicalNodes.get(selectedId);
  }
  return nodes.get(selectedId) || null;
}

function renderAnalytics(node) {
  const liveNode = nodes.get(node.id) || node;
  const cls = stClass(selectedDate ? liveNode : node);

  el.panelPlace.textContent = node.nama;
  el.camChip.textContent = "CAM " +
    String([...nodes.keys()].indexOf(node.id) + 1).padStart(2, "0");
  el.camChip.className = "cam-chip " + cls;

  renderStreamHealth();

  el.sumTotal.textContent = fmtNum(total(node));
  el.sumMotor.textContent = fmtNum(node.motor);
  el.sumMobil.textContent = fmtNum(node.mobil);
  el.sumTruk.textContent = fmtNum(node.truk);

  el.countMotor.textContent = fmtNum(node.motor);
  el.countMobil.textContent = fmtNum(node.mobil);
  el.countTruk.textContent = fmtNum(node.truk);

  el.avgHour.textContent = fmtNum(averagePerHour(node));
  el.avgBasis.textContent = node.avgBasisLabel ||
    (selectedDate ? "rata-rata 24 jam" : "dari 0.2 jam aktif");
  el.peakHour.textContent = selectedDate
    ? "-"
    : (node.startTime ? node.peakOccupancy : 0);

  el.chartSub.textContent = selectedDate
    ? "24 jam \u2014 " + formatShortDateId(selectedDate) + " (WIB)"
    : "24 jam (00:00 - 23:00 WIB)";

  renderChart(node);
  renderGauge(liveNode);
}

function averagePerHour(node) {
  if (node.avgPerHour != null) return node.avgPerHour;
  if (selectedDate) return Math.round(total(node) / 24);
  const elapsed = node.startTime ? Math.max((Date.now() / 1000 - node.startTime) / 3600, 0) : 0;
  const slots = node.slotCounts || {};
  const activeSlots = Object.values(slots).filter((b) => (Number(b && b.total) || 0) > 0).length;
  const activeHours = Math.max(elapsed, activeSlots * 0.25, 0.25);
  return total(node) > 0 ? Math.round(total(node) / activeHours) : 0;
}

function chartData(node) {
  const raw = node.hourlyCounts || {};
  const cm = Number(node.motor) || 0;
  const cb = Number(node.mobil) || 0;
  const ct = Number(node.truk) || 0;
  const csum = cm + cb + ct;

  const hours = Array.from({ length: 24 }, (_, h) => String(h).padStart(2, "0"));
  return hours.map((hour) => {
    const b = raw[hour];
    let motor = 0;
    let mobil = 0;
    let truk = 0;
    let tot = 0;

    if (b != null && typeof b === "object") {
      motor = Number(b.motor) || 0;
      mobil = Number(b.mobil) || 0;
      truk = Number(b.truk) || 0;
      tot = Number(b.total) || motor + mobil + truk;
    } else {
      tot = Number(b || 0);
    }

    const sum = motor + mobil + truk;
    if (tot > sum) {
      const gap = tot - sum;
      if (csum > 0) {
        const gm = Math.floor((gap * cm) / csum);
        const gb = Math.floor((gap * cb) / csum);
        motor += gm;
        mobil += gb;
        truk += Math.max(0, gap - gm - gb);
      } else {
        motor += gap;
      }
    }
    return { hour, motor, mobil, truk, total: tot };
  });
}

const CLASS_META = [
  { key: "motor", label: "Motor" },
  { key: "mobil", label: "Mobil" },
  { key: "truk", label: "Truk/Bus" },
];

const SEG_ORDER = ["motor", "mobil", "truk"];

let chartKey = "";

function chartBarHtml(p) {
  const segs = SEG_ORDER.map((k) =>
    '<i class="seg seg-' + k + '" data-cls="' + k + '"></i>').join("");
  const rows = CLASS_META.map((m) =>
    '<span class="tip-row"><i class="dot-' + m.key + '"></i>' + m.label +
    '<b data-v="' + m.key + '">0</b></span>').join("");
  return (
    '<div class="chart-bar" data-hour="' + p.hour + '" tabindex="0" role="img">' +
    '<div class="chart-stack">' + segs + "</div>" +
    '<div class="chart-tip" role="tooltip">' +
    '<div class="tip-head"><span class="tip-name"></span>' +
    '<span class="tip-total"></span></div>' +
    '<div class="tip-breakdown">' + rows + "</div>" +
    "</div>" +
    "</div>"
  );
}

function paintChart(points, peak) {
  const bars = el.chart.querySelectorAll(".chart-bar");
  const pct = (v) => (peak > 0 ? (v / peak) * 100 : 0);

  points.forEach((p, i) => {
    const bar = bars[i];
    if (!bar) return;

    let topKey = null;
    SEG_ORDER.forEach((k) => { if (p[k] > 0) topKey = k; });

    SEG_ORDER.forEach((k) => {
      const seg = bar.querySelector('[data-cls="' + k + '"]');
      if (!seg) return;
      seg.style.height = pct(p[k]) + "%";
      seg.classList.toggle("is-top", k === topKey);
    });

    bar.classList.toggle("is-peak", peak > 0 && p.total === peak);
    bar.classList.toggle("is-selected", p.hour === selectedHour);
    bar.classList.toggle("tip-left", i < 5);
    bar.classList.toggle("tip-right", i > 18);
    bar.setAttribute("aria-pressed", p.hour === selectedHour ? "true" : "false");
    bar.querySelector(".tip-name").textContent = p.hour + ":00 WIB";
    bar.querySelector(".tip-total").textContent = fmtNum(p.total) + " Kendaraan";
    CLASS_META.forEach((m) => {
      bar.querySelector('[data-v="' + m.key + '"]').textContent = fmtNum(p[m.key]);
    });
    bar.setAttribute("aria-label",
      p.hour + ":00, total " + fmtNum(p.total) + " kendaraan");
  });
}

function formatSlotInterval(slotKey, sessionStarts) {
  const [hh, mm] = slotKey.split(":").map(Number);
  const startMin = hh * 60 + mm;
  const endMin = startMin + 15;
  const endStr =
    String(Math.floor(endMin / 60) % 24).padStart(2, "0") +
    ":" +
    String(endMin % 60).padStart(2, "0");

  let matched = null;
  (sessionStarts || []).forEach((raw) => {
    const parts = String(raw || "").trim().split(":");
    if (parts.length !== 2) return;
    const sh = Number(parts[0]);
    const sm = Number(parts[1]);
    if (!Number.isFinite(sh) || !Number.isFinite(sm)) return;
    const sMin = sh * 60 + sm;
    if (sMin >= startMin && sMin < endMin && (!matched || sMin < matched.min)) {
      matched = {
        min: sMin,
        label: String(sh).padStart(2, "0") + ":" + String(sm).padStart(2, "0"),
      };
    }
  });
  return matched
    ? matched.label + " (system start) - " + endStr
    : slotKey + " - " + endStr;
}

function defaultDrilldownHour(points, node) {
  for (let i = points.length - 1; i >= 0; i--) {
    if (points[i].total > 0) return points[i].hour;
  }
  const slotHours = Object.keys(node.slotCounts || {}).map((k) => k.slice(0, 2)).sort();
  if (slotHours.length) return slotHours[slotHours.length - 1];
  const starts = (node.sessionStarts || []).map((s) => s.slice(0, 2)).sort();
  if (starts.length) return starts[starts.length - 1];
  if (!selectedDate) {
    return new Date().toLocaleTimeString("en-GB", {
      timeZone: "Asia/Jakarta",
      hour: "2-digit",
    }).slice(0, 2);
  }
  return "00";
}

function renderSlotDetail(node, points) {
  if (!selectedHour || !/^\d{2}$/.test(selectedHour)) {
    selectedHour = defaultDrilldownHour(points, node);
  }

  el.slotHourSelect.innerHTML = points.map((p) => {
    const suffix = p.total > 0 ? " (" + fmtNum(p.total) + ")" : "";
    return '<option value="' + p.hour + '"' +
      (p.hour === selectedHour ? " selected" : "") + ">" +
      p.hour + ":00 - " + p.hour + ":59" + suffix + "</option>";
  }).join("");
  el.slotHourSelect.value = selectedHour;

  const slots = node.slotCounts || {};
  const starts = node.sessionStarts || [];
  let sumM = 0;
  let sumC = 0;
  let sumT = 0;
  let sumTot = 0;

  const rows = [0, 15, 30, 45].map((m) => {
    const skey = selectedHour + ":" + String(m).padStart(2, "0");
    const b = slots[skey] || {};
    const motor = Number(b.motor) || 0;
    const mobil = Number(b.mobil) || 0;
    const truk = Number(b.truk) || 0;
    const tot = Math.max(Number(b.total) || 0, motor + mobil + truk);
    sumM += motor;
    sumC += mobil;
    sumT += truk;
    sumTot += tot;

    const label = formatSlotInterval(skey, starts);
    const hasStart = label.includes("(system start)");
    const cls = [tot > 0 ? "is-active" : "", hasStart ? "is-start" : ""]
      .filter(Boolean).join(" ");
    return (
      '<tr class="' + cls + '">' +
      "<td>" + label + "</td>" +
      "<td>" + fmtNum(motor) + "</td>" +
      "<td>" + fmtNum(mobil) + "</td>" +
      "<td>" + fmtNum(truk) + "</td>" +
      "<td><b>" + fmtNum(tot) + "</b></td>" +
      "</tr>"
    );
  });
  el.slotDetailBody.innerHTML = rows.join("");

  if (el.slotDetailFoot) {
    el.slotDetailFoot.innerHTML =
      "<tr>" +
        "<td>Total Jam " + selectedHour + ":00</td>" +
        '<td class="ft-motor">' + fmtNum(sumM) + "</td>" +
        '<td class="ft-mobil">' + fmtNum(sumC) + "</td>" +
        '<td class="ft-truk">' + fmtNum(sumT) + "</td>" +
        '<td class="ft-total">' + fmtNum(sumTot) + "</td>" +
      "</tr>";
  }
}

function selectChartHour(hour) {
  if (!hour || hour === selectedHour) return;
  selectedHour = hour;
  const node = currentAnalyticsNode();
  if (node) renderChart(node);
}

el.chart.addEventListener("click", (event) => {
  const bar = event.target.closest(".chart-bar");
  if (bar && bar.dataset.hour) selectChartHour(bar.dataset.hour);
});

el.chart.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  const bar = event.target.closest(".chart-bar");
  if (bar && bar.dataset.hour) {
    event.preventDefault();
    selectChartHour(bar.dataset.hour);
  }
});

el.slotHourSelect.addEventListener("change", () => {
  selectChartHour(el.slotHourSelect.value);
});

function renderChart(node) {
  const points = chartData(node);
  const peak = points.length ? Math.max(...points.map((p) => p.total)) : 0;
  const key = points.map((p) => p.hour).join(",");

  if (key !== chartKey) {
    chartKey = key;
    el.chart.innerHTML = points.length
      ? points.map(chartBarHtml).join("")
      : '<span class="chart-empty">Belum ada data per jam.</span>';
    el.chartAxis.innerHTML = points.map((p, i) => {
      const showTick = (i % 3 === 0) || i === 23;
      return "<span>" + (showTick ? p.hour : "") + "</span>";
    }).join("");
  }

  renderSlotDetail(node, points);
  paintChart(points, peak);

  el.chart.setAttribute(
    "aria-label",
    "Diagram batang bertumpuk jumlah kendaraan 24 jam: motor, mobil, truk/bus. " +
    "Puncak " + peak + " kendaraan."
  );
}

function renderGauge(node) {
  const cls = stClass(node);
  const pct = Math.max(0, Math.min(100, node.occupancy));

  el.gauge.className = "gauge " + cls;
  el.gaugeFill.style.width = pct + "%";
  el.occValue.textContent = pct + "%";

  el.gaugeStatus.className = "gauge-status occ-badge " + cls;
  el.occLabel.textContent = rawStatus(node);
  el.occNote.textContent = NOTES[node.status] || "";
}

/* ---- Dynamic node registry -------------------------------------- */

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
      zones: incoming.zones || (prev && prev.zones) || [],
      motor: (prev && prev.motor) || 0,
      mobil: (prev && prev.mobil) || 0,
      truk: (prev && prev.truk) || 0,
      occupancy: (prev && prev.occupancy) || 0,
      online: prev ? prev.online : false,
      status: (prev && prev.status) || "OFFLINE",
      streamStatus: (prev && prev.streamStatus) || null,
      staleSeconds: (prev && prev.staleSeconds) || null,
      startTime: (prev && prev.startTime) || 0,
      peakOccupancy: (prev && prev.peakOccupancy) || 0,
      hourlyCounts: (prev && prev.hourlyCounts) || {},
      slotCounts: (prev && prev.slotCounts) || {},
      sessionStarts: (prev && prev.sessionStarts) || [],
      avgPerHour: prev ? prev.avgPerHour : null,
      avgBasisLabel: (prev && prev.avgBasisLabel) || "",
    });
  });
}

function mapNodePayload(id, incoming, prev) {
  return {
    id: id,
    nama: incoming.nama || (prev && prev.nama) || id,
    lat: incoming.lat != null ? incoming.lat : (prev && prev.lat),
    lng: incoming.lng != null ? incoming.lng : (prev && prev.lng),
    zones: incoming.zones || (prev && prev.zones) || [],
    motor: incoming.motor || 0,
    mobil: incoming.mobil || 0,
    truk: incoming.truk || 0,
    occupancy: incoming.occupancy || 0,
    online: incoming.online != null
      ? Boolean(incoming.online) : (prev && prev.online) || false,
    status: incoming.status || (prev && prev.status) || "OFFLINE",
    streamStatus: incoming.stream_status ||
      (prev && prev.streamStatus) ||
      (incoming.online ? "LIVE" : "OFFLINE"),
    staleSeconds: incoming.stale_seconds != null
      ? Number(incoming.stale_seconds)
      : (prev && prev.staleSeconds) || null,
    startTime: incoming.start_time != null
      ? Number(incoming.start_time) : (prev && prev.startTime) || 0,
    peakOccupancy: incoming.peak_occupancy != null
      ? Number(incoming.peak_occupancy) : (prev && prev.peakOccupancy) || 0,
    hourlyCounts: incoming.hourly_counts != null
      ? incoming.hourly_counts : (prev && prev.hourlyCounts) || {},
    slotCounts: incoming.slot_counts != null
      ? incoming.slot_counts : (prev && prev.slotCounts) || {},
    sessionStarts: incoming.session_starts != null
      ? incoming.session_starts : (prev && prev.sessionStarts) || [],
    avgPerHour: incoming.avg_per_hour != null ? Number(incoming.avg_per_hour) : null,
    avgBasisLabel: incoming.avg_basis_label || "",
  };
}

/* ---- Live polling ----------------------------------------------- */

function applyNodes(apiNodes) {
  Object.entries(apiNodes).forEach(([id, incoming]) => {
    nodes.set(id, mapNodePayload(id, incoming, nodes.get(id)));
  });

  upsertMarkers();
  renderMenu();
  renderPreviewCard();
  if (selectedId && !selectedDate && nodes.has(selectedId)) {
    renderAnalytics(nodes.get(selectedId));
  }
  renderStreamHealth();
}

async function poll() {
  try {
    const res = await fetch(API_BASE + "/traffic-data", { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const payload = await res.json();
    if (payload.date) todayWibDate = payload.date;
    applyNodes(payload.nodes || {});
    if (Date.now() - lastDatesFetchAt > 30000) loadHistoryDates();
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
    renderStreamHealth();
  } catch (err) {
    console.warn("[ATCS] initial render failed:", err.message);
  }

  poll();
  setInterval(poll, POLL_MS);
}

boot();
