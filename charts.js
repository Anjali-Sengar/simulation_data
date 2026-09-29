"use strict";
// Stages 3-4: sensor trend, Digital Twin vs Actual, health trend. No calculations here - only plotting API data.
const COL = { actual: "#38d0ff", twin: "#f5a623", resid: "#ff6b9d" };
const HCOL = { HEALTHY: "#2ecc71", WARNING: "#f5a623", ALERT: "#ff4d4f", TRANSIENT: "#4aa3ff" };
const panels = [];
let cur = 0, refreshTimer = null;

const cursorPlugin = { id: "cursorLine", afterDatasetsDraw(ch) {
  if (ch.$pos == null) return;
  const x = ch.scales.x.getPixelForValue(ch.$pos), a = ch.chartArea, c = ch.ctx;
  c.save(); c.strokeStyle = "rgba(255,255,255,.6)"; c.lineWidth = 1; c.setLineDash([4, 3]);
  c.beginPath(); c.moveTo(x, a.top); c.lineTo(x, a.bottom); c.stroke(); c.restore();
} };
const nearest = (arr, v) => { let lo = 0, hi = arr.length - 1;
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < v) lo = m + 1; else hi = m; }
  return (lo > 0 && Math.abs(arr[lo - 1] - v) < Math.abs(arr[lo] - v)) ? lo - 1 : lo; };
const hms = t => typeof t === "string" ? t.slice(11, 19) : String(t ?? "");

function opts(yTitle, yExtra = {}) {
  const tick = { color: "#8fa0bf" }, grid = { color: "#1b2745" };
  return { responsive: true, maintainAspectRatio: false, animation: false,
    interaction: { mode: "index", intersect: false },
    plugins: { legend: { labels: { color: "#8fa0bf" } } },
    scales: { x: { ticks: { ...tick, maxTicksLimit: 8, autoSkip: true }, grid },
              y: { title: { display: !!yTitle, text: yTitle, color: "#8fa0bf" }, ticks: tick, grid, ...yExtra } } };
}
const line = (label, data, color, extra = {}) =>
  ({ label, data, borderColor: color, borderWidth: 1.6, pointRadius: 0, spanGaps: false, ...extra });

function mk(p, slot, canvasId, cfg) {
  if (p[slot]) p[slot].destroy();
  p[slot] = new Chart($(canvasId), { type: "line", plugins: [cursorPlugin], ...cfg });
}

function buildTrend(p) {
  const d = p.data;
  mk(p, "c1", p.canvas, { data: { labels: d.t.map(hms), datasets: [line(`${d.label} actual`, d.actual, COL.actual)] },
                          options: opts(`${d.label} (${d.unit})`) });
}
function buildTwin(p) {
  const d = p.data;
  $(p.msgId).textContent = d.expected ? "" : "Digital Twin expected values unavailable (raw dataset not merged).";
  mk(p, "c1", p.canvas, { data: { labels: d.t.map(hms), datasets: [
      line("Actual", d.actual, COL.actual),
      line("Digital Twin (expected)", d.expected || [], COL.twin, { borderDash: [6, 4] })] },
    options: opts(`${d.label} (${d.unit})`) });
  const o = opts(`Residual (${d.unit})`); o.plugins.legend.display = false;
  mk(p, "c2", p.canvas2, { data: { labels: d.t.map(hms), datasets: [
      line("Residual", d.residual, COL.resid, { fill: "origin", backgroundColor: "rgba(255,107,157,.12)" })] }, options: o });
}
function buildHealth(p) {
  const d = p.data;
  mk(p, "c1", p.canvas, { data: { labels: d.t.map(hms), datasets: [
      line("Health index", d.health_index, "#8fa0bf", { segment: { borderColor: c => HCOL[d.health_status[c.p1DataIndex]] || "#8fa0bf" } })] },
    options: { ...opts("Health index (0–100)", { min: 0, max: 100 }),
               plugins: { legend: { display: false }, tooltip: { callbacks: { afterLabel: c => d.health_status[c.dataIndex] || "" } } } } });
}

function span(range) {
  const n = state.meta.rows;
  if (range === "all") return [0, n];
  const half = Math.max(30, Math.round(Number(range) / (state.meta.dt_seconds || 1)));
  return [Math.max(0, cur - half), Math.min(n, cur + half + 1)];
}
async function refresh(p) {
  if (p.kind !== "health" && !p.sensor) return;
  const [s, e] = span($(p.rangeId).value);
  const key = `${p.kind}|${p.sensor}|${s}|${e}`;
  if (key !== p.key) {
    try {
      const url = p.kind === "health" ? `/api/health_series?start=${s}&end=${e}`
                                      : `/api/series?sensor=${p.sensor}&start=${s}&end=${e}`;
      const r = await fetch(url); if (!r.ok) throw new Error(r.status);
      p.data = await r.json(); p.key = key; p.build(p);
      if (p.kind !== "twin") $(p.msgId).textContent = "";
    } catch (err) { $(p.msgId).textContent = "data error: " + err.message; return; }
  }
  const pos = nearest(p.data.index, cur);
  for (const c of [p.c1, p.c2]) if (c) { c.$pos = pos; c.draw(); }
}
function selectSensor(p, k) {
  p.sensor = k;
  document.querySelectorAll(`#${p.tabsId} .tab`).forEach(b => b.classList.toggle("on", b.dataset.k === k));
  refresh(p);
}
function makePanel(cfg) {
  const p = { key: null, data: null, ...cfg }; panels.push(p);
  if (p.tabsId) {
    const box = $(p.tabsId);
    state.meta.sensors.forEach(s => { const b = document.createElement("button");
      b.className = "tab"; b.textContent = s.label; b.dataset.k = s.key; b.onclick = () => selectSensor(p, s.key); box.appendChild(b); });
  }
  $(p.rangeId).addEventListener("change", () => refresh(p));
  if (p.sensor) selectSensor(p, p.sensor);
}

document.addEventListener("meta-ready", () => {
  if (typeof Chart === "undefined") {
    document.querySelectorAll(".chartbox").forEach(b => b.textContent = "Chart.js could not load (internet needed for the CDN).");
    return;
  }
  makePanel({ kind: "trend", tabsId: "trendTabs", rangeId: "trendRange", canvas: "trendCanvas", msgId: "trendMsg",
              sensor: state.meta.sensors[0].key, build: buildTrend });
  makePanel({ kind: "twin", tabsId: "twinTabs", rangeId: "twinRange", canvas: "twinCanvas", canvas2: "residCanvas",
              msgId: "twinMsg", sensor: null, build: buildTwin });        // defaults to top sensor on first update
  makePanel({ kind: "health", rangeId: "healthRange", canvas: "healthCanvas", msgId: "healthMsg", build: buildHealth });
});
document.addEventListener("cursor", e => {
  if (!panels.length) return;
  cur = e.detail.index;
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(() => {
    const twin = panels.find(p => p.kind === "twin");
    if (twin && !twin.sensor) {
      const top = e.detail.engine.top_sensor, ok = state.meta.sensors.some(s => s.key === top);
      selectSensor(twin, ok ? top : state.meta.sensors[0].key);
    }
    panels.forEach(refresh);
  }, 120);
});
