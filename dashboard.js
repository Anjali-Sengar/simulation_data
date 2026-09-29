"use strict";
const $ = id => document.getElementById(id);
const state = { meta: null, timer: null };

const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(d);
const pretty = s => s ? s.replace(/_/g, " ").replace(/^\w/, c => c.toUpperCase()) : "—";
const faultName = k => k == null ? "—" : (state.meta.labels.faults[k] || pretty(k));
const sensorLabel = k => { const s = state.meta.sensors.find(x => x.key === k); return s ? s.label : pretty(k); };

const HEALTH_CLS = { HEALTHY: "ok", WARNING: "warn", ALERT: "bad", TRANSIENT: "trans" };
const LEVEL_CLS = ["ok", "warn", "bad", "bad"];
const clsOf = (map, k) => map[k] || "na";

function paintCard(id, cls) {
  const el = $(id); el.classList.remove("b-ok", "b-warn", "b-bad", "b-trans"); el.classList.add("b-" + cls);
}
function setStatus(el, text, cls) {
  el.textContent = text ?? "—";
  el.className = el.className.replace(/\bs-\w+/g, "").trim() + " s-" + cls;
}

function render(d) {
  const e = d.engine;
  // health
  const cls = clsOf(HEALTH_CLS, e.health_status);
  $("hiVal").textContent = num(e.health_index, 0);
  setStatus($("hiStatus"), e.health_status, cls);
  const bar = $("hiBar"); bar.style.width = Math.max(0, Math.min(100, e.health_index ?? 0)) + "%";
  bar.style.background = `var(--${cls === "na" ? "mute" : cls})`;
  paintCard("cardHealth", cls);

  // anomaly (derived only from existing engine-level level column)
  const lvl = e.engine_max_status_level;
  const detected = lvl !== null && lvl >= 2;
  const acls = lvl === null ? "na" : (detected ? "bad" : lvl === 1 ? "warn" : "ok");
  const aText = e.in_transient ? "TRANSIENT" : detected ? "DETECTED" : lvl === 1 ? "WATCH" : lvl === 0 ? "NONE" : "—";
  setStatus($("anStatus"), aText, e.in_transient ? "trans" : acls);
  paintCard("cardAnom", e.in_transient ? "trans" : acls);
  $("mlScore").textContent = e.ml_score === null ? "N/A (transient)" : num(e.ml_score, 3);
  $("mlLabel").textContent = e.ml_label ?? "—";
  $("nAff").textContent = e.n_anomalous === undefined ? "—" : `${e.n_anomalous} anomalous / ${e.n_watch} watch+`;
  $("topSensor").textContent = e.top_sensor ? sensorLabel(e.top_sensor) : "—";

  // fault diagnosis
  const normal = e.predicted_fault === "normal_operation";
  setStatus($("predFault"), faultName(e.predicted_fault), e.predicted_fault == null ? "na" : normal ? "ok" : "warn");
  $("predConf").textContent = "Confidence " + (e.predicted_confidence == null ? "—" : (e.predicted_confidence * 100).toFixed(1) + " %");
  const sev = e.gt_fault_severity ? ` · severity ${num(e.gt_fault_severity, 2)}` : "";
  $("gtFault").textContent = e.gt_fault_type == null ? "not available" : faultName(e.gt_fault_type) + sev;
  paintCard("cardFault", e.predicted_fault == null ? "trans" : normal ? "ok" : "warn");

  // sensors
  const grid = $("sensorGrid"); grid.innerHTML = "";
  for (const [k, s] of Object.entries(d.sensors)) {
    const c = LEVEL_CLS[s.status_level] ?? "na";
    const div = document.createElement("div");
    div.className = "sensor"; div.style.borderLeftColor = `var(--${c === "na" ? "mute" : c})`;
    const name = state.meta.labels.status_levels[String(s.status_level)] ?? "—";
    div.innerHTML = `<div class="n"></div><div class="v"><span></span> <small></small></div>
      <div class="m"><span class="z"></span><span class="st"></span></div>`;
    div.querySelector(".n").textContent = s.label;
    div.querySelector(".v span").textContent = num(s.actual, k.includes("pressure") || k.includes("vibration") || k.includes("voltage") ? 2 : 1);
    div.querySelector(".v small").textContent = s.unit;
    div.querySelector(".z").textContent = "z " + num(s.z, 2);
    const st = div.querySelector(".st"); st.textContent = name; st.className = "st s-" + c;
    grid.appendChild(div);
  }

  // current state
  const rows = [
    ["Operating condition", pretty(e.operating_condition)],
    ["Health", `${num(e.health_index, 1)} · ${e.health_status ?? "—"}`],
    ["Anomaly state", aText],
    ["Predicted fault", faultName(e.predicted_fault)],
    ["Affected sensors", e.n_anomalous === undefined ? "—" : `${e.n_anomalous} anomalous, ${e.n_watch} watch+`],
    ["In transient", e.in_transient === null ? "—" : e.in_transient ? "Yes" : "No"],
    ["Data timestamp (UTC)", d.timestamp ? d.timestamp.replace("T", " ").slice(0, 19) : "—"],
  ];
  $("stateGrid").innerHTML = "";
  for (const [a, b] of rows) {
    const div = document.createElement("div"), sp = document.createElement("span");
    sp.textContent = a; div.appendChild(sp); div.appendChild(document.createTextNode(b));
    $("stateGrid").appendChild(div);
  }
  $("ts").textContent = `row ${d.index + 1} / ${state.meta.rows} · ` + (d.timestamp ? d.timestamp.replace("T", " ").slice(0, 19) + " UTC" : "—");
}

async function load(index) {
  try {
    const r = await fetch("/api/latest" + (index == null ? "" : "?index=" + index));
    if (!r.ok) throw new Error(r.status);
    const d = await r.json(); render(d);
    document.dispatchEvent(new CustomEvent("cursor", { detail: d }));
  } catch (err) { $("ts").textContent = "data error: " + err.message; }
}

async function init() {
  try {
    state.meta = await (await fetch("/api/meta")).json();
  } catch (err) { $("srcMode").textContent = "backend unreachable"; return; }
  $("srcMode").textContent = state.meta.source_mode;
  const sl = $("slider"); sl.max = state.meta.rows - 1; sl.value = state.meta.default_index;
  sl.addEventListener("input", () => { clearTimeout(state.timer); state.timer = setTimeout(() => load(sl.value), 60); });
  $("btnLatest").addEventListener("click", () => { sl.value = sl.max; load(sl.value); });
  document.dispatchEvent(new Event("meta-ready"));
  load(sl.value);
}
init();
