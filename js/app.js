const SOURCE_LABELS = {
  weight: "Weight",
  grad: "Gradient",
  update: "Update",
  activation: "Activation",
  preactivation: "Pre-Activation",
  gelu_activation: "GELU / Massive Act",
  attention: "Attention",
  logits: "Logits",
};

const SOURCE_COLORS = {
  weight: "#9fd4e4",
  grad: "#8fd9a4",
  update: "#e8c888",
  activation: "#f0a8b6",
  preactivation: "#e0b0c4",
  gelu_activation: "#b8d4f0",
  attention: "#7ecfc4",
  logits: "#e0c08a",
};

const SOURCE_KIND_ORDER = [
  "weight",
  "grad",
  "update",
  "activation",
  "preactivation",
  "gelu_activation",
  "attention",
  "logits",
];

const ARCH_NODES = {
  embeddings: [
    { id: "wte", title: "Token Embed", subtitle: "wte", path: "transformer.wte" },
    { id: "wpe", title: "Pos Embed", subtitle: "wpe", path: "transformer.wpe" },
  ],
  head: [
    { id: "ln_f", title: "Final LN", subtitle: "ln_f", path: "transformer.ln_f" },
    { id: "lm_head", title: "LM Head", subtitle: "lm_head", path: "lm_head", optional: true },
  ],
};

const LAYER_COLORS = [
  "#9fd4e4", "#f0a8b6", "#8fd9a4", "#e8c888", "#b8d4f0", "#7ecfc4",
  "#e0b0c4", "#b8e08a", "#f0d080", "#a8d8f0", "#90d0c4", "#e0c08a",
];

const RUN_COLORS = [
  "#9fd4e4", "#e8c888", "#8fd9a4", "#f0a8b6", "#b8d4f0", "#7ecfc4",
];

const REFERENCE_LINE_COLORS = ["#f0d080", "#f0a8b6", "#8fd9a4", "#b8d4f0", "#7ecfc4"];

// Logarithmic axes cannot represent zero. Keep step 0 separate from the
// genuine step 1 instead of mapping both to x=1.
const LOG_ZERO_PLOT_X = 0.1;
const LOG_SCALE_NON_POSITIVE_MESSAGE =
  "The selected step range contains zero or negative values, so a logarithmic y-axis cannot be plotted.";

function plotXForStep(step, logarithmic) {
  return logarithmic && step === 0 ? LOG_ZERO_PLOT_X : step;
}

let manifest = null;
let runId = null;
let availableRuns = [];
let manifestCache = new Map();
let lossLogByRun = new Map();
let compareRuns = false;
let compareLossRuns = false;
let setupResidualMode = false;
let selectedRuns = new Set();
let runsLoading = false;
let manifestLoadGen = 0;
let specById = new Map();
let specsByModule = new Map();
let specsByFamily = new Map();
let activeLayer = 0;
let activeModuleId = null;
let activeSpecId = null;
let compareLayers = false;
let selectedLayers = new Set();
let chart = null;
let lossChart = null;
let lossLog = null;
let lossViewMode = "both";
/** @type {"linear" | "loglinear" | "loglog"} */
let lossScaleMode = "linear";
/** @type {"linear" | "loglinear" | "loglog"} */
let curveScaleMode = "linear";
const CURVE_LOGLOG_ENABLED = true;
/** @type {"step" | "tau"} */
let xAxisMode = "step";
let lossStepMin = null;
let lossStepMax = null;
let fullChart = null;
let fullOverlayOpen = false;
let fullOverlayMode = null;
let definitionResizeObserver = null;
let referenceLines = [];
let referenceLineDrag = null;
let referenceLineDragCanvas = null;
let referenceLineSuppressClick = false;
/** Snapshot of main-page compare state while fullscreen is open. */
let fullscreenCompareSnapshot = null;

function displaySpecLabel(spec) {
  return typeof observableDisplayLabel === "function"
    ? observableDisplayLabel(spec)
    : spec?.label || spec?.id || "Observable";
}

function snapshotCompareState() {
  return {
    compareRuns,
    compareLossRuns,
    setupResidualMode,
    selectedRuns: new Set(selectedRuns),
  };
}

function restoreCompareState(snap) {
  if (!snap) return;
  compareRuns = !!snap.compareRuns;
  compareLossRuns = !!snap.compareLossRuns;
  setupResidualMode = !!snap.setupResidualMode;
  selectedRuns = new Set(snap.selectedRuns || []);
  const runInput = document.getElementById("compareRuns");
  if (runInput) runInput.checked = compareRuns;
  const lossInput = document.getElementById("compareLossRuns");
  if (lossInput) lossInput.checked = compareLossRuns;
  const curveRes = document.getElementById("curveResidual");
  if (curveRes) curveRes.checked = setupResidualMode && compareRuns;
  const lossRes = document.getElementById("lossResidual");
  if (lossRes) lossRes.checked = setupResidualMode && compareLossRuns;
  const spec = activeSpec();
  updateCompareToggle(spec);
  updateLossCompareToggle();
  updateResidualToggles();
  renderRunPicker();
  if (spec) renderChart(spec);
  renderLossChart();
}

async function boot() {
  runId = new URLSearchParams(location.search).get("run");
  if (!runId) {
    window.location.href = "select.html";
    return;
  }

  try {
    if (window.Chart && window.ChartZoom) {
      try { Chart.register(window.ChartZoom); } catch (_) { /* already registered */ }
    }
    const annPlugin =
      window["chartjs-plugin-annotation"] ||
      window.ChartAnnotation ||
      window.annotationPlugin;
    if (window.Chart && annPlugin) {
      try { Chart.register(annPlugin); } catch (_) { /* already registered */ }
    }
    manifest = await fetchJson(`data/${runId}/manifest.json`);
    const visibleSpecs = (manifest.specs || []).filter(specHasCurveData);
    specById = new Map(visibleSpecs.map((s) => [s.id, s]));
    specsByModule = groupSpecsByModule(visibleSpecs);
    specsByFamily = groupSpecsByFamily(visibleSpecs);
    manifestCache.set(runId, { specById, model: manifest.model });
    await loadAvailableRuns();
    wireEvents();
    if (typeof wireNotesUi === "function") wireNotesUi();
    await renderHeader();
    lossLog = await loadLossLog();
    lossLogByRun.set(runId, lossLog);
    syncLossRangeInputs();
    updateLossCompareToggle();
    renderLossChart();
    renderLayerTabs();
    renderArchitecture();
    if (typeof refreshNotes === "function") refreshNotes();
  } catch (err) {
    document.getElementById("architecture").innerHTML =
      `<div class="error">Failed to load viewer data: ${escapeHtml(err.message)}<br><br>` +
      `Run <code>python3 scripts/build_viewer_data.py</code> or ` +
      `<a href="select.html">choose another dataset</a>.</div>`;
  }
}

function wireEvents() {
  document.getElementById("compareLayers").addEventListener("change", (e) => {
    compareLayers = e.target.checked;
    const spec = activeSpec();
    if (compareLayers) {
      compareRuns = false;
      const runInput = document.getElementById("compareRuns");
      if (runInput) runInput.checked = false;
      if (spec) ensureDefaultSelectedLayers(spec);
    }
    if (spec) {
      updateCompareToggle(spec);
      renderChart(spec);
    } else {
      updateCompareToggle(null);
    }
  });
  document.getElementById("compareRuns")?.addEventListener("change", async (e) => {
    await setCompareRuns(e.target.checked);
  });
  document.getElementById("compareRunsFull")?.addEventListener("change", async (e) => {
    await setCompareRuns(e.target.checked);
  });
  document.getElementById("compareLossRunsFull")?.addEventListener("change", async (e) => {
    await setCompareLossRuns(e.target.checked);
  });
  document.getElementById("curveResidual")?.addEventListener("change", (e) => {
    setSetupResidualMode(e.target.checked);
  });
  document.getElementById("lossResidual")?.addEventListener("change", (e) => {
    setSetupResidualMode(e.target.checked);
  });
  document.getElementById("fullResidual")?.addEventListener("change", (e) => {
    setSetupResidualMode(e.target.checked);
  });
  document.getElementById("fullPickAllRuns")?.addEventListener("click", async () => {
    selectedRuns = new Set(availableRuns.map((r) => r.run_id));
    renderRunPicker();
    renderFullRunPicker();
    await ensureSelectedRunManifests();
    refreshCompareViews();
  });
  document.getElementById("fullPickCurrentRun")?.addEventListener("click", async () => {
    selectedRuns = new Set([runId]);
    renderRunPicker();
    renderFullRunPicker();
    refreshCompareViews();
  });
  document.getElementById("pickAllRuns")?.addEventListener("click", async () => {
    selectedRuns = new Set(availableRuns.map((r) => r.run_id));
    renderRunPicker();
    renderFullRunPicker();
    await ensureSelectedRunManifests();
    const spec = activeSpec();
    if (spec) renderChart(spec);
    refreshCompareViews();
  });
  document.getElementById("pickCurrentRun")?.addEventListener("click", () => {
    selectedRuns = new Set([runId]);
    renderRunPicker();
    renderFullRunPicker();
    const spec = activeSpec();
    if (spec) renderChart(spec);
    refreshCompareViews();
  });
  document.getElementById("resetZoomBtn").addEventListener("click", resetChartZoom);
  document.getElementById("expandBtn").addEventListener("click", openFullscreen);
  document.getElementById("fullCloseBtn").addEventListener("click", closeFullscreen);
  document.getElementById("fullResetZoomBtn").addEventListener("click", () => {
    safeResetFullChartZoom();
  });
  document.getElementById("addReferenceLineBtn")?.addEventListener("click", addReferenceLine);
  document.getElementById("clearReferenceLinesBtn")?.addEventListener("click", clearReferenceLines);
  document.getElementById("referenceSlope")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") addReferenceLine();
  });
  document.getElementById("chartOverlay").addEventListener("click", (e) => {
    if (e.target.id === "chartOverlay") closeFullscreen();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || !fullOverlayOpen) return;
    const noteModal = document.getElementById("noteModal");
    if (noteModal && !noteModal.hidden) return;
    if (typeof isNotesRailExpanded === "function" && isNotesRailExpanded()) return;
    closeFullscreen();
  });
  document.getElementById("pickCurrentLayer").addEventListener("click", () => {
    const spec = activeSpecId ? specById.get(activeSpecId) : null;
    if (!spec) return;
    selectedLayers = new Set([activeLayer]);
    renderLayerPicker(spec);
    renderChart(spec);
  });
  document.getElementById("pickAllLayers").addEventListener("click", () => {
    const spec = activeSpecId ? specById.get(activeSpecId) : null;
    if (!spec) return;
    selectedLayers = new Set(familyMembers(spec).map((m) => m.layer));
    renderLayerPicker(spec);
    renderChart(spec);
  });
  document.getElementById("pickClearLayers").addEventListener("click", () => {
    const spec = activeSpecId ? specById.get(activeSpecId) : null;
    selectedLayers = new Set();
    if (spec) {
      renderLayerPicker(spec);
      renderChart(spec);
    }
  });

  document.getElementById("lossResetZoomBtn")?.addEventListener("click", () => {
    renderLossChart();
    if (fullOverlayOpen && fullOverlayMode === "loss") renderFullLossChart();
  });
  document.getElementById("lossExpandBtn")?.addEventListener("click", openLossFullscreen);
  document.getElementById("compareLossRuns")?.addEventListener("change", async (e) => {
    await setCompareLossRuns(e.target.checked);
  });
  document.querySelectorAll(".loss-view-btn").forEach((btn) => {
    btn.addEventListener("click", () => setLossViewMode(btn.dataset.mode));
  });
  document.querySelectorAll(".loss-scale-btn").forEach((btn) => {
    btn.addEventListener("click", () => setLossScaleMode(btn.dataset.scale));
  });
  document.querySelectorAll(".curve-scale-btn").forEach((btn) => {
    btn.addEventListener("click", () => setCurveScaleMode(btn.dataset.scale));
  });
  document.querySelectorAll(".x-axis-btn").forEach((btn) => {
    btn.addEventListener("click", () => setXAxisMode(btn.dataset.xAxis));
  });
  document.getElementById("lossApplyRangeBtn")?.addEventListener("click", applyLossStepRange);
  document.getElementById("lossResetRangeBtn")?.addEventListener("click", resetLossStepRange);
  document.getElementById("lossApplyRangeFullBtn")?.addEventListener("click", () => {
    const minEl = document.getElementById("lossStepMinFull");
    const maxEl = document.getElementById("lossStepMaxFull");
    const mainMin = document.getElementById("lossStepMin");
    const mainMax = document.getElementById("lossStepMax");
    if (mainMin && minEl) mainMin.value = minEl.value;
    if (mainMax && maxEl) mainMax.value = maxEl.value;
    applyLossStepRange();
  });
  for (const id of ["lossStepMin", "lossStepMax", "lossStepMinFull", "lossStepMaxFull"]) {
    document.getElementById(id)?.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        if (id.endsWith("Full")) document.getElementById("lossApplyRangeFullBtn")?.click();
        else applyLossStepRange();
      }
    });
  }
}

async function loadAvailableRuns() {
  try {
    const data = await fetchJson("data/index.json");
    availableRuns = Array.isArray(data.runs) ? data.runs : [];
  } catch (_) {
    availableRuns = [{ run_id: runId, label: runId }];
  }
  if (!availableRuns.some((r) => r.run_id === runId)) {
    availableRuns = [{ run_id: runId, label: runId }, ...availableRuns];
  }
  selectedRuns = new Set(availableRuns.map((r) => r.run_id));
}

function canCompareRuns() {
  return availableRuns.length >= 2;
}

function runLabel(rid) {
  const found = availableRuns.find((r) => r.run_id === rid);
  return found?.label || rid;
}

function runColor(rid) {
  const idx = availableRuns.findIndex((r) => r.run_id === rid);
  return RUN_COLORS[(idx >= 0 ? idx : 0) % RUN_COLORS.length];
}

function activeSpec() {
  return activeSpecId ? specById.get(activeSpecId) : null;
}

async function ensureRunManifest(rid) {
  const cached = manifestCache.get(rid);
  if (cached && !cached.error) return cached;
  try {
    const m = await fetchJson(`data/${rid}/manifest.json`);
    const map = new Map((m.specs || []).filter(specHasCurveData).map((s) => [s.id, s]));
    const entry = { specById: map, model: m.model };
    manifestCache.set(rid, entry);
    return entry;
  } catch (err) {
    // Do not sticky-cache failures — allow retry after rebuild / rename.
    manifestCache.delete(rid);
    throw err;
  }
}

async function ensureSelectedRunManifests() {
  const ids = [...selectedRuns].filter((rid) => rid !== runId);
  if (!ids.length) return;
  const gen = ++manifestLoadGen;
  runsLoading = true;
  setRunPickStatus("Loading other setups… (large manifests, may take a few seconds)");
  try {
    await Promise.all(ids.map((rid) => ensureRunManifest(rid)));
    if (xAxisMode === "tau") {
      await Promise.all(ids.map((rid) => ensureLossLogForRun(rid)));
    }
    if (gen !== manifestLoadGen) return;
    setRunPickStatus("");
  } catch (err) {
    if (gen !== manifestLoadGen) return;
    setRunPickStatus(`Failed to load setup data: ${err.message}`, true);
  } finally {
    if (gen === manifestLoadGen) runsLoading = false;
  }
}

function setRunPickStatus(text, isError = false) {
  const el = document.getElementById("runPickStatus");
  if (!el) return;
  if (!text) {
    el.hidden = true;
    el.textContent = "";
    el.classList.remove("is-error");
    return;
  }
  el.hidden = false;
  el.textContent = text;
  el.classList.toggle("is-error", !!isError);
}

async function setCompareRuns(on) {
  compareRuns = on;
  if (on) {
    compareLayers = false;
    const layerInput = document.getElementById("compareLayers");
    if (layerInput) layerInput.checked = false;
    if (!selectedRuns.size) {
      selectedRuns = new Set(availableRuns.map((r) => r.run_id));
    }
    await ensureSelectedRunManifests();
    if (setupResidualMode) await ensureBaselineManifest();
  } else if (!compareLossRuns) {
    setupResidualMode = false;
  }
  const spec = activeSpec();
  updateCompareToggle(spec);
  updateResidualToggles();
  updateFullscreenCompareChrome();
  if (spec) renderChart(spec);
  if (fullOverlayOpen && fullOverlayMode === "spec" && spec) renderFullChart(spec);
}

async function ensureLossLogForRun(rid) {
  const cached = lossLogByRun.get(rid);
  if (cached && !cached.error) return cached;
  const text = await fetchText(`data/${rid}/eval_loss_log.csv`);
  if (!text) {
    lossLogByRun.delete(rid);
    return { error: true };
  }
  const parsed = parseLossCsv(text);
  if (!parsed) {
    lossLogByRun.delete(rid);
    return { error: true };
  }
  lossLogByRun.set(rid, parsed);
  return parsed;
}

function selectedRunIds() {
  return [...selectedRuns];
}

/** Prefer the run whose id/label is "baseline"; else current run; else first selected. */
function resolveBaselineRunId(selectedIds = selectedRunIds()) {
  const ids = selectedIds.length ? selectedIds : availableRuns.map((r) => r.run_id);
  if (ids.includes("baseline")) return "baseline";
  const labeled = availableRuns.find(
    (r) => ids.includes(r.run_id) && /^baseline$/i.test(String(r.label || "").trim())
  );
  if (labeled) return labeled.run_id;
  const soft = availableRuns.find(
    (r) => ids.includes(r.run_id) && /baseline/i.test(String(r.label || r.run_id))
  );
  if (soft) return soft.run_id;
  // Residual mode still needs baseline data even if not checked — look globally.
  if (availableRuns.some((r) => r.run_id === "baseline")) return "baseline";
  const globalSoft = availableRuns.find((r) => /baseline/i.test(String(r.label || r.run_id)));
  if (globalSoft) return globalSoft.run_id;
  if (ids.includes(runId)) return runId;
  return ids[0] || null;
}

async function ensureBaselineManifest() {
  const bid = resolveBaselineRunId();
  if (!bid) return;
  await ensureRunManifest(bid);
  await ensureLossLogForRun(bid);
}

function pointStepKey(p) {
  if (p && Number.isFinite(p._step)) return p._step;
  if (p && Number.isFinite(p.x)) return p.x;
  return null;
}

/** y_other − y_baseline at matching steps (keeps plot-x / _step from other). */
function residualAgainstBaseline(points, baselinePoints) {
  const baseMap = new Map();
  for (const p of baselinePoints || []) {
    const key = pointStepKey(p);
    if (key == null || !Number.isFinite(p.y)) continue;
    baseMap.set(key, p.y);
  }
  const out = [];
  for (const p of points || []) {
    const key = pointStepKey(p);
    if (key == null || !Number.isFinite(p.y) || !baseMap.has(key)) continue;
    out.push({
      x: p.x,
      y: p.y - baseMap.get(key),
      _step: Number.isFinite(p._step) ? p._step : key,
      _axisX: p._axisX,
      ...(Number.isFinite(p._tau) ? { _tau: p._tau } : {}),
    });
  }
  return out;
}

function zeroBaselinePoints(baselinePoints) {
  return (baselinePoints || []).map((p) => ({
    x: p.x,
    y: 0,
    _step: Number.isFinite(p._step) ? p._step : pointStepKey(p),
    _axisX: p._axisX,
    ...(Number.isFinite(p._tau) ? { _tau: p._tau } : {}),
  }));
}

function setSetupResidualMode(on) {
  setupResidualMode = !!on;
  updateResidualToggles();
  if (setupResidualMode) {
    ensureBaselineManifest().then(() => {
      const spec = activeSpec();
      if (spec && compareRuns) renderChart(spec);
      if (compareLossRuns) {
        renderLossChart();
        if (fullOverlayOpen && fullOverlayMode === "loss") renderFullLossChart();
      }
      if (fullOverlayOpen && fullOverlayMode === "spec" && spec) renderFullChart(spec);
    });
  } else {
    const spec = activeSpec();
    if (spec && compareRuns) renderChart(spec);
    if (compareLossRuns) {
      renderLossChart();
      if (fullOverlayOpen && fullOverlayMode === "loss") renderFullLossChart();
    }
    if (fullOverlayOpen && fullOverlayMode === "spec" && spec) renderFullChart(spec);
  }
}

function updateResidualToggles() {
  for (const [wrapId, inputId, active] of [
    ["curveResidualWrap", "curveResidual", compareRuns && canCompareRuns()],
    ["lossResidualWrap", "lossResidual", compareLossRuns && canCompareRuns()],
  ]) {
    const wrap = document.getElementById(wrapId);
    const input = document.getElementById(inputId);
    if (!wrap || !input) continue;
    wrap.hidden = !active;
    if (!active) {
      // keep setupResidualMode if the other compare mode still needs it
      input.checked = false;
    } else {
      input.checked = setupResidualMode;
    }
  }
  if (!compareRuns && !compareLossRuns) setupResidualMode = false;
  updateFullscreenCompareChrome();
}

async function setCompareLossRuns(on) {
  compareLossRuns = on;
  if (on) {
    await Promise.all(selectedRunIds().map((rid) => ensureLossLogForRun(rid)));
    if (setupResidualMode) await ensureBaselineManifest();
  } else if (!compareRuns) {
    setupResidualMode = false;
  }
  updateLossCompareToggle();
  updateResidualToggles();
  updateFullscreenCompareChrome();
  syncLossRangeInputs();
  renderLossChart();
  if (fullOverlayOpen && fullOverlayMode === "loss") renderFullLossChart();
}

function updateLossCompareToggle() {
  const wrap = document.getElementById("compareLossRunsWrap");
  const input = document.getElementById("compareLossRuns");
  if (!wrap || !input) return;
  const ok = canCompareRuns() && lossLog && !lossLog.error;
  wrap.hidden = !ok;
  if (!ok) {
    compareLossRuns = false;
    input.checked = false;
  } else {
    input.checked = compareLossRuns;
  }
  updateResidualToggles();
}

async function fetchJson(url) {
  const sep = url.includes("?") ? "&" : "?";
  const res = await fetch(`${url}${sep}t=${Date.now()}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${url} (${res.status})`);
  return res.json();
}

async function fetchText(url) {
  const sep = url.includes("?") ? "&" : "?";
  const res = await fetch(`${url}${sep}t=${Date.now()}`, { cache: "no-store" });
  if (!res.ok) return null;
  return res.text();
}

function specHasCurveData(spec) {
  const series = spec?.series;
  const hasSeries =
    Array.isArray(series?.steps) &&
    Array.isArray(series?.values) &&
    series.steps.length > 0 &&
    series.values.length > 0;
  return hasSeries || Boolean(spec?.curve_png);
}

function groupSpecsByModule(specs) {
  const map = new Map();
  for (const spec of specs) {
    if (!map.has(spec.ui_module)) map.set(spec.ui_module, []);
    map.get(spec.ui_module).push(spec);
  }
  return map;
}

function groupSpecsByFamily(specs) {
  const map = new Map();
  for (const spec of specs) {
    if (!spec.family_id) continue;
    if (!map.has(spec.family_id)) map.set(spec.family_id, []);
    map.get(spec.family_id).push(spec);
  }
  return map;
}

async function resolveRunLabel() {
  try {
    const data = await fetchJson("data/index.json");
    const run = (data.runs || []).find((r) => r.run_id === runId);
    return run?.label || runId;
  } catch (_) {
    return runId;
  }
}

async function renderHeader() {
  const label = await resolveRunLabel();
  document.getElementById("runSubtitle").textContent =
    `${specById.size} observables`;
  document.getElementById("runBadge").textContent = label;
  document.title = `${label} · nanoGPT Observable Explorer`;
}

function renderLayerTabs() {
  const tabs = document.getElementById("layerTabs");
  tabs.innerHTML = "";
  for (let i = 0; i < manifest.model.n_layer; i += 1) {
    const btn = document.createElement("button");
    btn.className = `layer-tab${i === activeLayer ? " active" : ""}`;
    btn.textContent = `L${i}`;
    btn.addEventListener("click", () => {
      activeLayer = i;
      renderLayerTabs();
      renderArchitecture();
      clearSelection();
    });
    tabs.appendChild(btn);
  }
  document.getElementById("layerBadge").textContent = `Block ${activeLayer}`;
}

function moduleIdForBlock(suffix) {
  return `h.${activeLayer}.${suffix}`;
}

function renderArchitecture() {
  const root = document.getElementById("architecture");
  root.innerHTML = "";

  const flow = document.createElement("div");
  flow.className = "flow";

  flow.appendChild(flowLabel("Input Tokens"));
  flow.appendChild(connector());

  const embRow = document.createElement("div");
  embRow.className = "node-row";
  for (const node of ARCH_NODES.embeddings) {
    embRow.appendChild(createModuleNode(node.id, node.title, node.subtitle, node.id));
  }
  flow.appendChild(embRow);
  flow.appendChild(connector());
  flow.appendChild(flowLabel(`Transformer Block ${activeLayer}`));

  const blockGrid = document.createElement("div");
  blockGrid.className = "block-grid";

  const attnCol = section("Attention");
  attnCol.appendChild(createModuleNode(moduleIdForBlock("ln_1"), "LayerNorm", "ln_1", moduleIdForBlock("ln_1")));
  attnCol.appendChild(createModuleNode(moduleIdForBlock("attn.c_attn"), "QKV Proj", "c_attn", moduleIdForBlock("attn.c_attn")));
  attnCol.appendChild(createModuleNode(moduleIdForBlock("attn"), "Attention", "attn · entropy", moduleIdForBlock("attn")));
  attnCol.appendChild(createModuleNode(moduleIdForBlock("attn.c_proj"), "Attn Out", "c_proj", moduleIdForBlock("attn.c_proj")));

  const mlpCol = section("MLP");
  mlpCol.appendChild(createModuleNode(moduleIdForBlock("ln_2"), "LayerNorm", "ln_2", moduleIdForBlock("ln_2")));
  mlpCol.appendChild(createModuleNode(moduleIdForBlock("mlp.c_fc"), "MLP Up", "c_fc", moduleIdForBlock("mlp.c_fc")));
  mlpCol.appendChild(createModuleNode(moduleIdForBlock("mlp.gelu"), "GELU", "gelu · massive", moduleIdForBlock("mlp.gelu")));
  mlpCol.appendChild(createModuleNode(moduleIdForBlock("mlp.c_proj"), "MLP Down", "c_proj", moduleIdForBlock("mlp.c_proj")));

  const sideCol = section("Residual");
  const res1 = document.createElement("div");
  res1.className = "residual";
  res1.appendChild(noteNode("Add & Norm"));
  const res2 = document.createElement("div");
  res2.className = "residual";
  res2.appendChild(noteNode("Add & Norm"));
  sideCol.appendChild(res1);
  sideCol.appendChild(res2);

  blockGrid.appendChild(attnCol);
  blockGrid.appendChild(sideCol);
  blockGrid.appendChild(mlpCol);
  flow.appendChild(blockGrid);

  flow.appendChild(connector());
  flow.appendChild(flowLabel("Output Head"));

  const headRow = document.createElement("div");
  headRow.className = "node-row";
  for (const node of ARCH_NODES.head) {
    headRow.appendChild(createModuleNode(node.id, node.title, node.subtitle, node.id, node.optional));
  }
  flow.appendChild(headRow);

  root.appendChild(flow);
}

function section(label) {
  const wrap = document.createElement("div");
  wrap.className = "block-section";
  const title = document.createElement("div");
  title.className = "block-section-label";
  title.textContent = label;
  wrap.appendChild(title);
  return wrap;
}

function flowLabel(text) {
  const el = document.createElement("div");
  el.className = "flow-label";
  el.textContent = text;
  return el;
}

function connector() {
  const el = document.createElement("div");
  el.className = "connector";
  return el;
}

function noteNode(text) {
  const el = document.createElement("div");
  el.className = "module-node disabled";
  el.innerHTML = `<div class="title">${text}</div><div class="subtitle">skip connection</div>`;
  return el;
}

function createModuleNode(uiModuleId, title, subtitle, moduleKey, optional = false) {
  const specs = specsByModule.get(moduleKey) || [];
  const count = specs.length;
  const el = document.createElement("button");
  el.type = "button";
  el.className = "module-node";
  if (count > 0) {
    el.classList.add("clickable");
    if (activeModuleId === moduleKey) el.classList.add("active");
    el.addEventListener("click", () => selectModule(moduleKey, title, subtitle));
  } else {
    el.classList.add("disabled");
    el.disabled = true;
  }

  el.innerHTML = `
    ${count > 0 ? `<span class="count">${count}</span>` : ""}
    <div class="title">${escapeHtml(title)}</div>
    <div class="subtitle">${escapeHtml(subtitle)}${count === 0 ? " · no data" : ""}</div>
  `;
  return el;
}

function friendlyModuleLabel(moduleKey, fallback) {
  const raw = manifest.modules[moduleKey]?.label;
  if (raw && raw !== moduleKey) return raw;
  const mAttn = /^h\.(\d+)\.attn$/.exec(moduleKey);
  if (mAttn) return `Block ${mAttn[1]} · Attention (entropy / sink)`;
  const mGelu = /^h\.(\d+)\.mlp\.gelu$/.exec(moduleKey);
  if (mGelu) return `Block ${mGelu[1]} · MLP GELU (massive activation)`;
  if (moduleKey === "lm_head") return "LM Head (logits)";
  return raw || fallback || moduleKey;
}

function selectModule(moduleKey, title, subtitle) {
  activeModuleId = moduleKey;
  activeSpecId = null;
  renderArchitecture();

  const specs = specsByModule.get(moduleKey) || [];
  document.getElementById("detailEmpty").style.display = "none";
  document.getElementById("detailContent").classList.add("visible");

  document.getElementById("moduleTitle").textContent = friendlyModuleLabel(moduleKey, title);
  const exampleSelector = specs[0]?.selector || `transformer.${moduleKey}`;
  document.getElementById("modulePath").textContent = exampleSelector;

  // Auto-selects first spec via renderSpecGroups → selectSpec → renderChart.
  // Do not call renderChart(null) afterward (that blanked the chart).
  renderSpecGroups(specs);
}

function renderSpecGroups(specs) {
  const groups = {};
  for (const spec of specs) {
    groups[spec.source_kind] = groups[spec.source_kind] || [];
    groups[spec.source_kind].push(spec);
  }

  const container = document.getElementById("specGroups");
  container.innerHTML = "";

  const kinds = [
    ...SOURCE_KIND_ORDER,
    ...Object.keys(groups).filter((k) => !SOURCE_KIND_ORDER.includes(k)),
  ];
  for (const kind of kinds) {
    const items = groups[kind];
    if (!items?.length) continue;

    const group = document.createElement("div");
    group.className = "spec-group";
    group.innerHTML = `<div class="spec-group-title">${SOURCE_LABELS[kind] || kind}</div>`;

    const list = document.createElement("div");
    list.className = "spec-list";
    for (const spec of items) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = `spec-chip${spec.id === activeSpecId ? " active" : ""}`;
      chip.textContent = displaySpecLabel(spec);
      chip.addEventListener("click", () => {
        activeSpecId = spec.id;
        renderSpecGroups(specs);
        renderChart(spec);
      });
      list.appendChild(chip);
    }
    group.appendChild(list);
    container.appendChild(group);
  }

  if (!activeSpecId && specs.length) {
    selectSpec(specs[0]);
    renderSpecGroups(specs);
  }
}

function selectSpec(spec) {
  activeSpecId = spec.id;
  renderChart(spec);
}

function familyMembers(spec) {
  if (!spec?.family_id) return [];
  const members = specsByFamily.get(spec.family_id) || [];
  return members.filter(
    (s) => s.layer !== null && s.layer !== undefined && s.series?.steps?.length
  );
}

function canCompareLayers(spec) {
  return familyMembers(spec).length >= 2;
}

function availableLayers(spec) {
  return familyMembers(spec).map((m) => m.layer).sort((a, b) => a - b);
}

function ensureDefaultSelectedLayers(spec) {
  const available = new Set(availableLayers(spec));
  selectedLayers = new Set([...selectedLayers].filter((l) => available.has(l)));
  if (selectedLayers.size === 0) {
    if (available.has(activeLayer)) selectedLayers.add(activeLayer);
    else if (available.size) selectedLayers.add([...available][0]);
  }
}

function updateCompareToggle(spec) {
  const layerWrap = document.getElementById("compareToggleWrap");
  const layerInput = document.getElementById("compareLayers");
  const layerPickWrap = document.getElementById("layerPickWrap");
  const runWrap = document.getElementById("compareRunsWrap");
  const runInput = document.getElementById("compareRuns");
  const runPickWrap = document.getElementById("runPickWrap");

  const canLayers = !!(spec && canCompareLayers(spec));
  const canRunsGlobal = canCompareRuns();
  const showRunToggle = !!(spec && canRunsGlobal);

  // Hide toggles when no spec — but do NOT reset compareRuns / compareLayers.
  if (runWrap) runWrap.hidden = !showRunToggle;
  if (runInput) runInput.checked = compareRuns;
  if (runPickWrap) {
    if (showRunToggle && compareRuns) {
      renderRunPicker();
      runPickWrap.hidden = false;
    } else {
      runPickWrap.hidden = true;
    }
  }

  if (compareRuns && compareLayers) {
    compareLayers = false;
  }

  if (layerWrap) layerWrap.hidden = !canLayers || compareRuns;
  if (!canLayers || compareRuns) {
    if (layerInput) layerInput.checked = false;
    if (layerPickWrap) layerPickWrap.hidden = true;
    updateResidualToggles();
    return;
  }
  if (layerInput) layerInput.checked = compareLayers;
  if (compareLayers) {
    ensureDefaultSelectedLayers(spec);
    renderLayerPicker(spec);
    if (layerPickWrap) layerPickWrap.hidden = false;
  } else if (layerPickWrap) {
    layerPickWrap.hidden = true;
  }
  updateResidualToggles();
}

function refreshCompareViews() {
  const spec = activeSpec();
  if (spec && compareRuns) renderChart(spec);
  if (compareLossRuns) renderLossChart();
  if (fullOverlayOpen && fullOverlayMode === "loss") renderFullLossChart();
  if (fullOverlayOpen && fullOverlayMode === "spec" && spec) renderFullChart(spec);
  updateFullscreenCompareChrome();
}

function renderRunPicker() {
  const list = document.getElementById("runPickList");
  if (!list) return;
  list.innerHTML = "";
  for (const run of availableRuns) {
    const color = runColor(run.run_id);
    const checked = selectedRuns.has(run.run_id);
    const label = document.createElement("label");
    label.className = `layer-pick-chip${checked ? " active" : ""}`;
    label.style.setProperty("--chip-color", color);
    const name = run.label || run.run_id;
    const currentMark = run.run_id === runId ? " · current" : "";
    label.innerHTML = `
      <input type="checkbox" ${checked ? "checked" : ""} />
      <span class="layer-pick-dot"></span>
      <span>${escapeHtml(name)}${currentMark}</span>
    `;
    label.querySelector("input").addEventListener("change", async (e) => {
      if (e.target.checked) selectedRuns.add(run.run_id);
      else selectedRuns.delete(run.run_id);
      renderRunPicker();
      renderFullRunPicker();
      await ensureSelectedRunManifests();
      const spec = activeSpec();
      if (spec) renderChart(spec);
      if (compareLossRuns) {
        await setCompareLossRuns(true);
      }
      updateFullscreenCompareChrome();
    });
    list.appendChild(label);
  }
}

function renderFullRunPicker() {
  const list = document.getElementById("fullRunPickList");
  if (!list) return;
  list.innerHTML = "";
  for (const run of availableRuns) {
    const color = runColor(run.run_id);
    const checked = selectedRuns.has(run.run_id);
    const label = document.createElement("label");
    label.className = `layer-pick-chip${checked ? " active" : ""}`;
    label.style.setProperty("--chip-color", color);
    const name = run.label || run.run_id;
    const currentMark = run.run_id === runId ? " · current" : "";
    label.innerHTML = `
      <input type="checkbox" ${checked ? "checked" : ""} />
      <span class="layer-pick-dot"></span>
      <span>${escapeHtml(name)}${currentMark}</span>
    `;
    label.querySelector("input").addEventListener("change", async (e) => {
      if (e.target.checked) selectedRuns.add(run.run_id);
      else selectedRuns.delete(run.run_id);
      renderRunPicker();
      renderFullRunPicker();
      await ensureSelectedRunManifests();
      if (compareLossRuns) await Promise.all(selectedRunIds().map((rid) => ensureLossLogForRun(rid)));
      refreshCompareViews();
    });
    list.appendChild(label);
  }
}

function updateFullscreenCompareChrome() {
  const lossWrap = document.getElementById("compareLossRunsFullWrap");
  const lossInput = document.getElementById("compareLossRunsFull");
  const curveWrap = document.getElementById("compareRunsFullWrap");
  const curveInput = document.getElementById("compareRunsFull");
  const residualWrap = document.getElementById("fullResidualWrap");
  const residualInput = document.getElementById("fullResidual");
  const pick = document.getElementById("fullRunPick");

  const inLoss = fullOverlayOpen && fullOverlayMode === "loss";
  const inSpec = fullOverlayOpen && fullOverlayMode === "spec";
  const can = canCompareRuns();

  if (lossWrap) lossWrap.hidden = !(inLoss && can && lossLog && !lossLog.error);
  if (lossInput) lossInput.checked = !!(inLoss && compareLossRuns);

  if (curveWrap) curveWrap.hidden = !(inSpec && can);
  if (curveInput) curveInput.checked = !!(inSpec && compareRuns);

  const residualActive =
    (inLoss && compareLossRuns && can) || (inSpec && compareRuns && can);
  if (residualWrap) residualWrap.hidden = !residualActive;
  if (residualInput) residualInput.checked = residualActive && setupResidualMode;

  const showPick =
    (inLoss && compareLossRuns && can) || (inSpec && compareRuns && can);
  if (pick) {
    pick.hidden = !showPick;
    if (showPick) renderFullRunPicker();
  }
}

function renderLayerPicker(spec) {
  const list = document.getElementById("layerPickList");
  list.innerHTML = "";
  for (const layer of availableLayers(spec)) {
    const color = LAYER_COLORS[layer % LAYER_COLORS.length];
    const label = document.createElement("label");
    label.className = `layer-pick-chip${selectedLayers.has(layer) ? " active" : ""}`;
    label.style.setProperty("--chip-color", color);
    label.innerHTML = `
      <input type="checkbox" ${selectedLayers.has(layer) ? "checked" : ""} />
      <span class="layer-pick-dot"></span>
      <span>L${layer}</span>
    `;
    label.querySelector("input").addEventListener("change", (e) => {
      if (e.target.checked) selectedLayers.add(layer);
      else selectedLayers.delete(layer);
      renderLayerPicker(spec);
      renderChart(spec);
    });
    list.appendChild(label);
  }
}

/** Collect unique plot-x values from chart datasets (log-x tick positions). */
function collectPlotXsFromChart(chart) {
  const xs = new Set();
  for (const ds of chart?.data?.datasets || []) {
    if (ds._referenceLine) continue;
    for (const p of ds.data || []) {
      if (p && Number.isFinite(p.x) && p.x > 0) xs.add(p.x);
    }
  }
  return [...xs].sort((a, b) => a - b);
}

/**
 * Prefer tidy steps: 1, 2, 5 × 10^n (e.g. 20000 over 19400). Lower is better.
 */
function stepUglyScore(v) {
  if (!(v > 0)) return 0;
  const exp = Math.floor(Math.log10(v) + 1e-12);
  const mant = v / Math.pow(10, exp);
  const d = Math.min(
    Math.abs(mant - 1),
    Math.abs(mant - 2),
    Math.abs(mant - 5),
    Math.abs(mant - 10)
  );
  const sig = String(Math.round(v)).replace(/0+$/, "").length;
  return d * 80 + Math.max(0, sig - 1) * 3;
}

function isTidyStep(v) {
  return stepUglyScore(v) < 8;
}

/**
 * Keep first/last; fill with tidy data steps spaced in log space.
 * Avoids dense late-axis clutter and prefers round labels like 20000.
 */
function thinLogAxisTicks(values, maxTicks = 8) {
  if (!values.length) return [];
  if (values.length <= 2) return values;

  const first = values[0];
  const last = values[values.length - 1];
  if (first === last) return [first];

  const middle = values.filter((v) => v !== first && v !== last);
  const tidy = middle.filter(isTidyStep).sort((a, b) => a - b);
  const pool = tidy.length ? tidy : middle;

  if (pool.length === 0) return [first, last];

  const innerSlots = Math.max(0, maxTicks - 2);
  if (pool.length <= innerSlots) {
    return [first, ...pool, last];
  }

  const picked = [];
  const logMin = Math.log(first);
  const logMax = Math.log(last);
  for (let i = 1; i <= innerSlots; i += 1) {
    const target = Math.exp(logMin + ((logMax - logMin) * i) / (innerSlots + 1));
    let best = null;
    let bestScore = Infinity;
    for (const v of pool) {
      if (picked.includes(v)) continue;
      // Prefer closeness in log space, then tidiness (20000 >> 19400).
      const dist = Math.abs(Math.log(v) - Math.log(target));
      const score = dist * 12 + (xAxisMode === "step" ? stepUglyScore(v) : 0);
      if (score < bestScore) {
        best = v;
        bestScore = score;
      }
    }
    if (best != null) picked.push(best);
  }
  picked.sort((a, b) => a - b);
  return [first, ...picked, last];
}

/**
 * Log-x ticks: only mark a sparse subset of steps present in the curve.
 * Plot x=1 stands for true step 0, so label it "0".
 */
function logXAxisTickConfig(axisColor) {
  return {
    color: axisColor,
    autoSkip: false,
    maxRotation: 45,
    minRotation: 0,
    callback(value) {
      const v = Number(value);
      if (!Number.isFinite(v)) return "";
      const isZeroSentinel = (this.chart?.data?.datasets || []).some(
        (ds) => !ds._referenceLine && (ds.data || []).some(
          (p) => p?._axisX === 0 && Math.abs(p.x - v) <= Math.max(1e-15, Math.abs(v) * 1e-10),
        ),
      );
      if (isZeroSentinel) return "0";
      return formatXAxisValue(v);
    },
  };
}

function afterBuildLogXTicks(axis) {
  const values = thinLogAxisTicks(collectPlotXsFromChart(axis.chart), 8);
  axis.ticks = values.map((value) => ({ value }));
}

function formatXAxisValue(value) {
  if (!Number.isFinite(value)) return "";
  if (xAxisMode === "step" && Math.abs(value - Math.round(value)) < 1e-9) {
    return String(Math.round(value));
  }
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e4)) return value.toExponential(2);
  return Number(value.toPrecision(4)).toString();
}

function xAxisTitle() {
  return xAxisMode === "tau" ? "τ = ∫ learning rate" : "Step";
}

function chartScaleOptions({
  scaleMode = "linear",
  xTitle = xAxisTitle(),
  yTitle = "Value",
  residual = false,
} = {}) {
  const axis = "rgba(255, 255, 255, 0.78)";
  const grid = "rgba(255, 255, 255, 0.12)";
  // Residuals cross zero, so y stays linear; x may still be logarithmic.
  const logX = scaleMode === "loglinear" || scaleMode === "loglog";
  const logY = !residual && scaleMode === "loglog";
  return {
    x: {
      type: logX ? "logarithmic" : "linear",
      title: { display: true, text: logX ? `${xTitle} (log)` : xTitle, color: axis },
      ticks: logX ? logXAxisTickConfig(axis) : { color: axis },
      ...(logX ? { afterBuildTicks: afterBuildLogXTicks } : {}),
      grid: { color: grid },
    },
    y: {
      type: logY ? "logarithmic" : "linear",
      title: {
        display: true,
        text: residual ? yTitle : logY ? `${yTitle} (log)` : yTitle,
        color: axis,
      },
      ticks: { color: axis },
      grid: { color: grid },
    },
  };
}

function zoomPluginOptions({ enablePan = true } = {}) {
  return {
    zoom: {
      wheel: { enabled: true, speed: 0.1 },
      pinch: { enabled: true },
      mode: "xy",
      drag: { enabled: false },
      onZoomComplete: ({ chart: zoomedChart }) => {
        if (zoomedChart === fullChart && referenceLines.length) {
          applyReferenceLinesToFullChart();
        }
      },
    },
    pan: {
      // Fullscreen charts disable pan so Hammer does not interfere with note clicks.
      enabled: enablePan,
      mode: "xy",
      modifierKey: enablePan ? "alt" : null,
      threshold: 10,
      onPanComplete: ({ chart: pannedChart }) => {
        if (pannedChart === fullChart && referenceLines.length) {
          applyReferenceLinesToFullChart();
        }
      },
    },
    limits: {
      x: { min: "original", max: "original", minRange: 1 },
      y: { min: "original", max: "original", minRange: 1e-12 },
    },
  };
}

function chartCommonOptions({
  legend = false,
  onClick = null,
  enablePan = true,
  scaleMode = "linear",
  yTitle = "Value",
  residual = false,
} = {}) {
  // Residuals can be negative: preserve log-x but never use log-y.
  const effectiveScale = residual && scaleMode === "loglog" ? "loglinear" : scaleMode;
  const opts = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "nearest", intersect: false },
    plugins: {
      legend: legend
        ? {
            display: true,
            position: "bottom",
            labels: { boxWidth: 12, color: "rgba(255, 255, 255, 0.85)", usePointStyle: true },
          }
        : { display: false },
      tooltip: {
        callbacks: {
          title: (items) => {
            if (!items.length) return "";
            const raw = items[0].raw;
            if (raw && Number.isFinite(raw._step)) {
              if (xAxisMode === "tau" && Number.isFinite(raw._tau)) {
                return `τ ${formatXAxisValue(raw._tau)} · step ${raw._step}`;
              }
              return `step ${raw._step}`;
            }
            return `step ${items[0].parsed.x}`;
          },
        },
      },
      zoom: zoomPluginOptions({ enablePan }),
      ...(residual
        ? {
            annotation: {
              annotations: {
                baselineZero: {
                  type: "line",
                  yMin: 0,
                  yMax: 0,
                  borderColor: "rgba(255, 255, 255, 0.35)",
                  borderWidth: 1,
                  borderDash: [4, 4],
                },
              },
            },
          }
        : {}),
    },
    scales: chartScaleOptions({
      scaleMode: effectiveScale,
      yTitle: residual ? "Δ vs baseline" : yTitle,
      residual,
    }),
  };
  if (typeof onClick === "function") opts.onClick = onClick;
  return opts;
}

function finiteMinMax(values) {
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return min === Infinity ? null : { min, max };
}

function referenceLineMode(chartInstance) {
  const logX = chartInstance?.scales?.x?.type === "logarithmic";
  const logY = chartInstance?.scales?.y?.type === "logarithmic";
  if (logX && logY) return "loglog";
  if (logX) return "loglinear";
  return "linear";
}

function referenceLinePoints(chartInstance, line) {
  const xScale = chartInstance?.scales?.x;
  const yScale = chartInstance?.scales?.y;
  const xMin = Number(xScale?.min);
  const xMax = Number(xScale?.max);
  const yMin = Number(yScale?.min);
  const yMax = Number(yScale?.max);
  if (![xMin, xMax, yMin, yMax].every(Number.isFinite) || !(xMax > xMin) || !(yMax > yMin)) {
    return null;
  }

  const mode = referenceLineMode(chartInstance);
  if (mode === "loglog") {
    if (!(xMin > 0) || !(yMin > 0)) return null;
    if (line.mode !== mode || !Number.isFinite(line.intercept)) {
      const xCenter = Math.sqrt(xMin * xMax);
      const yCenter = Math.sqrt(yMin * yMax);
      line.mode = mode;
      line.intercept = Math.log(yCenter) - line.slope * Math.log(xCenter);
    }
    const pointAt = (x) => {
      const logY = line.intercept + line.slope * Math.log(x);
      return { x, y: Math.exp(Math.max(-700, Math.min(700, logY))) };
    };
    return [pointAt(xMin), pointAt(xMax)];
  }

  if (mode === "loglinear") {
    if (!(xMin > 0)) return null;
    if (line.mode !== mode || !Number.isFinite(line.intercept)) {
      const xCenter = Math.sqrt(xMin * xMax);
      const yCenter = (yMin + yMax) / 2;
      line.mode = mode;
      line.intercept = yCenter - line.slope * Math.log10(xCenter);
    }
    const pointAt = (x) => ({ x, y: line.slope * Math.log10(x) + line.intercept });
    return [pointAt(xMin), pointAt(xMax)];
  }

  if (line.mode !== mode || !Number.isFinite(line.intercept)) {
    const xCenter = (xMin + xMax) / 2;
    const yCenter = (yMin + yMax) / 2;
    line.mode = mode;
    line.intercept = yCenter - line.slope * xCenter;
  }
  return [
    { x: xMin, y: line.slope * xMin + line.intercept },
    { x: xMax, y: line.slope * xMax + line.intercept },
  ];
}

function referenceLineDataset(chartInstance, line, index) {
  const points = referenceLinePoints(chartInstance, line);
  if (!points || !points.every((p) => Number.isFinite(p.x) && Number.isFinite(p.y))) return null;
  return {
    label: `Reference slope ${line.slope}`,
    data: points,
    borderColor: REFERENCE_LINE_COLORS[index % REFERENCE_LINE_COLORS.length],
    borderWidth: 2,
    borderDash: [8, 5],
    pointRadius: 0,
    pointHoverRadius: 0,
    tension: 0,
    fill: false,
    _referenceLine: true,
    _referenceLineIndex: index,
  };
}

function updateReferenceLineControls(message = "") {
  const clearBtn = document.getElementById("clearReferenceLinesBtn");
  if (clearBtn) clearBtn.hidden = referenceLines.length === 0;
  const list = document.getElementById("referenceLineList");
  if (list) {
    list.replaceChildren();
    referenceLines.forEach((line, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "reference-line-chip";
      button.title = `Remove reference line with slope ${line.slope}`;
      button.setAttribute("aria-label", button.title);
      button.addEventListener("click", () => removeReferenceLine(index));

      const swatch = document.createElement("span");
      swatch.className = "reference-line-swatch";
      swatch.style.color = REFERENCE_LINE_COLORS[index % REFERENCE_LINE_COLORS.length];
      const label = document.createElement("span");
      label.textContent = `m = ${line.slope}`;
      const remove = document.createElement("span");
      remove.className = "reference-line-remove";
      remove.textContent = "×";
      remove.setAttribute("aria-hidden", "true");
      button.append(swatch, label, remove);
      list.append(button);
    });
  }
  const status = document.getElementById("referenceLineStatus");
  if (status) {
    status.textContent = message ||
      (referenceLines.length ? `${referenceLines.length} line${referenceLines.length === 1 ? "" : "s"}` : "");
  }
}

function applyReferenceLinesToFullChart({ updateControls = true } = {}) {
  if (!fullChart) return;
  fullChart.data.datasets = (fullChart.data.datasets || []).filter((ds) => !ds._referenceLine);
  if (!referenceLines.length) {
    fullChart.update("none");
    if (updateControls) updateReferenceLineControls();
    return;
  }

  const xScale = fullChart.scales?.x;
  const yScale = fullChart.scales?.y;
  if (!xScale || !yScale) return;
  // Preserve the current viewport: steep reference lines should be clipped, not
  // force the user's data into a much smaller auto-scaled region.
  fullChart.options.scales.x.min = xScale.min;
  fullChart.options.scales.x.max = xScale.max;
  fullChart.options.scales.y.min = yScale.min;
  fullChart.options.scales.y.max = yScale.max;
  const datasets = referenceLines
    .map((line, index) => referenceLineDataset(fullChart, line, index))
    .filter(Boolean);
  fullChart.data.datasets.push(...datasets);
  fullChart.update("none");
  if (updateControls) updateReferenceLineControls();
}

function addReferenceLine() {
  const input = document.getElementById("referenceSlope");
  if (!input || !fullChart) return;
  if (!input.value.trim()) {
    updateReferenceLineControls("Enter a slope");
    input.focus();
    return;
  }
  const slope = Number(input.value);
  if (!Number.isFinite(slope)) {
    updateReferenceLineControls("Enter a finite slope");
    input.focus();
    return;
  }
  if (referenceLines.length >= 8) {
    updateReferenceLineControls("Maximum 8 lines");
    return;
  }
  referenceLines.push({ slope, intercept: null, mode: null });
  input.value = "";
  applyReferenceLinesToFullChart();
  const mode = referenceLineMode(fullChart);
  const modeLabel =
    mode === "loglog" ? "log–log" : mode === "loglinear" ? "log–linear" : "linear";
  updateReferenceLineControls(
    `Added m=${slope} (${modeLabel})`,
  );
}

function clearReferenceLines() {
  referenceLines = [];
  applyReferenceLinesToFullChart();
}

function removeReferenceLine(index) {
  if (!Number.isInteger(index) || index < 0 || index >= referenceLines.length) return;
  referenceLines.splice(index, 1);
  applyReferenceLinesToFullChart();
}

function resetReferenceLines() {
  referenceLines = [];
  referenceLineDrag = null;
  referenceLineSuppressClick = false;
  const input = document.getElementById("referenceSlope");
  if (input) input.value = "";
  updateReferenceLineControls();
}

function chartPointFromPointer(chartInstance, event) {
  const canvas = chartInstance?.canvas;
  if (!canvas) return null;
  const rect = canvas.getBoundingClientRect();
  if (!(rect.width > 0) || !(rect.height > 0)) return null;
  return {
    x: ((event.clientX - rect.left) / rect.width) * chartInstance.width,
    y: ((event.clientY - rect.top) / rect.height) * chartInstance.height,
  };
}

function pointToSegmentDistance(point, start, end) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  if (!(lengthSquared > 0)) return Math.hypot(point.x - start.x, point.y - start.y);
  const t = Math.max(
    0,
    Math.min(1, ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared),
  );
  return Math.hypot(point.x - (start.x + t * dx), point.y - (start.y + t * dy));
}

function referenceLineIndexAtPointer(event, threshold = 12) {
  if (!fullChart) return -1;
  const point = chartPointFromPointer(fullChart, event);
  const xScale = fullChart.scales?.x;
  const yScale = fullChart.scales?.y;
  if (!point || !xScale || !yScale) return -1;
  let bestIndex = -1;
  let bestDistance = threshold;
  for (const dataset of fullChart.data?.datasets || []) {
    if (!dataset._referenceLine || !Number.isInteger(dataset._referenceLineIndex)) continue;
    const points = dataset.data || [];
    if (points.length < 2) continue;
    const start = {
      x: xScale.getPixelForValue(points[0].x),
      y: yScale.getPixelForValue(points[0].y),
    };
    const end = {
      x: xScale.getPixelForValue(points[points.length - 1].x),
      y: yScale.getPixelForValue(points[points.length - 1].y),
    };
    if (![start.x, start.y, end.x, end.y].every(Number.isFinite)) continue;
    const distance = pointToSegmentDistance(point, start, end);
    if (distance <= bestDistance) {
      bestDistance = distance;
      bestIndex = dataset._referenceLineIndex;
    }
  }
  return bestIndex;
}

function onReferenceLinePointerDown(event) {
  if (event.button != null && event.button !== 0) return;
  const index = referenceLineIndexAtPointer(event);
  if (index < 0) return;
  referenceLineDrag = { index, pointerId: event.pointerId };
  referenceLineSuppressClick = true;
  if (typeof cancelPendingNoteClick === "function") cancelPendingNoteClick();
  try { event.currentTarget.setPointerCapture(event.pointerId); } catch (_) { /* optional */ }
  event.currentTarget.style.cursor = "grabbing";
  event.preventDefault();
  event.stopImmediatePropagation();
}

function onReferenceLinePointerMove(event) {
  if (!referenceLineDrag) {
    event.currentTarget.style.cursor =
      referenceLineIndexAtPointer(event) >= 0 ? "grab" : "crosshair";
    return;
  }
  if (event.pointerId !== referenceLineDrag.pointerId || !fullChart) return;
  const line = referenceLines[referenceLineDrag.index];
  const point = chartPointFromPointer(fullChart, event);
  const xScale = fullChart.scales?.x;
  const yScale = fullChart.scales?.y;
  if (!line || !point || !xScale || !yScale) return;
  const x = xScale.getValueForPixel(point.x);
  const y = yScale.getValueForPixel(point.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return;
  const mode = referenceLineMode(fullChart);
  if (mode === "loglog") {
    if (!(x > 0) || !(y > 0)) return;
    line.intercept = Math.log(y) - line.slope * Math.log(x);
  } else if (mode === "loglinear") {
    if (!(x > 0)) return;
    line.intercept = y - line.slope * Math.log10(x);
  } else {
    line.intercept = y - line.slope * x;
  }
  line.mode = mode;
  applyReferenceLinesToFullChart({ updateControls: false });
  event.preventDefault();
  event.stopImmediatePropagation();
}

function finishReferenceLineDrag(event, { suppressClick = true } = {}) {
  if (!referenceLineDrag || event.pointerId !== referenceLineDrag.pointerId) return;
  const line = referenceLines[referenceLineDrag.index];
  try { event.currentTarget.releasePointerCapture(event.pointerId); } catch (_) { /* optional */ }
  referenceLineDrag = null;
  referenceLineSuppressClick = suppressClick;
  if (suppressClick) {
    setTimeout(() => {
      referenceLineSuppressClick = false;
    }, 0);
  }
  event.currentTarget.style.cursor = "crosshair";
  updateReferenceLineControls(line ? `Moved m=${line.slope}` : "");
  event.preventDefault();
  event.stopImmediatePropagation();
}

function onReferenceLinePointerUp(event) {
  finishReferenceLineDrag(event);
}

function onReferenceLinePointerCancel(event) {
  finishReferenceLineDrag(event, { suppressClick: false });
}

function onReferenceLineClick(event) {
  if (!referenceLineSuppressClick) return;
  referenceLineSuppressClick = false;
  if (typeof cancelPendingNoteClick === "function") cancelPendingNoteClick();
  event.preventDefault();
  event.stopImmediatePropagation();
}

function detachReferenceLineDragHandlers() {
  const canvas = referenceLineDragCanvas;
  if (!canvas) return;
  canvas.removeEventListener("pointerdown", onReferenceLinePointerDown, true);
  canvas.removeEventListener("pointermove", onReferenceLinePointerMove, true);
  canvas.removeEventListener("pointerup", onReferenceLinePointerUp, true);
  canvas.removeEventListener("pointercancel", onReferenceLinePointerCancel, true);
  canvas.removeEventListener("click", onReferenceLineClick, true);
  canvas.style.cursor = "";
  referenceLineDragCanvas = null;
  referenceLineDrag = null;
}

function attachReferenceLineDragHandlers(canvas) {
  detachReferenceLineDragHandlers();
  if (!canvas) return;
  referenceLineDragCanvas = canvas;
  canvas.addEventListener("pointerdown", onReferenceLinePointerDown, true);
  canvas.addEventListener("pointermove", onReferenceLinePointerMove, true);
  canvas.addEventListener("pointerup", onReferenceLinePointerUp, true);
  canvas.addEventListener("pointercancel", onReferenceLinePointerCancel, true);
  canvas.addEventListener("click", onReferenceLineClick, true);
}

/** Restore axis range from data without destroying the Chart instance. */
function fitChartScalesToData(chart) {
  if (!chart?.data?.datasets?.length) return false;
  const xs = [];
  const ys = [];
  for (const ds of chart.data.datasets) {
    if (ds._referenceLine) continue;
    for (const p of ds.data || []) {
      if (p && Number.isFinite(p.x)) xs.push(p.x);
      if (p && Number.isFinite(p.y)) ys.push(p.y);
    }
  }
  if (!xs.length || !ys.length) return false;
  const xBounds = finiteMinMax(xs);
  const yBounds = finiteMinMax(ys);
  if (!xBounds || !yBounds) return false;
  const { min: xMin, max: xMax } = xBounds;
  const { min: yMin, max: yMax } = yBounds;
  const xLog = chart.options.scales?.x?.type === "logarithmic";
  const yLog = chart.options.scales?.y?.type === "logarithmic";
  if (xLog) {
    chart.options.scales.x.min = Math.max(xMin / 1.05, Number.EPSILON);
    chart.options.scales.x.max = xMax * 1.05;
  } else {
    const xPad = (xMax - xMin) * 0.01 || 1;
    chart.options.scales.x.min = xMin - xPad;
    chart.options.scales.x.max = xMax + xPad;
  }
  if (yLog) {
    chart.options.scales.y.min = Math.max(yMin / 1.08, Number.EPSILON);
    chart.options.scales.y.max = yMax * 1.08;
  } else {
    const yPad = (yMax - yMin) * 0.06 || Math.max(Math.abs(yMax) * 0.05, 1e-6);
    chart.options.scales.y.min = yMin - yPad;
    chart.options.scales.y.max = yMax + yPad;
  }
  try {
    chart.update("none");
    return true;
  } catch (_) {
    return false;
  }
}

function attachDoubleClickReset(canvas) {
  canvas.ondblclick = (evt) => {
    evt.preventDefault();
    resetChartZoom();
  };
}

/** Prefer rebuilding from data — resetZoom can leave linear scales in a dead state. */
function resetChartZoom() {
  const spec = activeSpec();
  if (spec) {
    renderChart(spec);
    return;
  }
  try {
    chart?.resetZoom?.();
  } catch (_) {
    /* ignore */
  }
}

function isNoteModalOpen() {
  const modal = document.getElementById("noteModal");
  return !!(modal && !modal.hidden);
}

/** Safe zoom reset: fit scales in-place. Avoid destroy/recreate (can blank the canvas). */
function safeResetFullChartZoom() {
  if (isNoteModalOpen()) return;
  if (!fullChart) {
    if (fullOverlayMode === "loss") renderFullLossChart();
    else {
      const spec = activeSpec();
      if (spec) renderFullChart(spec);
    }
    return;
  }
  try {
    fullChart.resetZoom?.();
  } catch (_) {
    /* ignore */
  }
  if (!fitChartScalesToData(fullChart)) {
    // Last resort: rebuild once
    if (fullOverlayMode === "loss") renderFullLossChart();
    else {
      const spec = activeSpec();
      if (spec) renderFullChart(spec);
    }
  } else {
    applyReferenceLinesToFullChart();
  }
}

function setChartChrome(hasSeriesChart) {
  document.getElementById("resetZoomBtn").hidden = !hasSeriesChart;
  document.getElementById("chartHint").hidden = !hasSeriesChart;
  document.getElementById("expandBtn").hidden = !hasSeriesChart;
  const scale = document.getElementById("curveScaleToggle");
  if (scale) scale.hidden = !hasSeriesChart || !CURVE_LOGLOG_ENABLED;
  const xAxis = document.getElementById("curveXAxisToggle");
  if (xAxis) xAxis.hidden = !hasSeriesChart;
  updateCurveScaleButtons();
  updateXAxisButtons();
}

function learningRateIntegralAt(log, step) {
  const points = log?.lr;
  if (!Array.isArray(points) || !points.length || !Number.isFinite(step)) return NaN;
  if (step <= 0) return 0;
  if (step <= points[0].x) return step * points[0].y;
  const last = points[points.length - 1];
  if (step >= last.x) return last.tau + (step - last.x) * last.y;

  let lo = 0;
  let hi = points.length - 1;
  while (lo + 1 < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (points[mid].x <= step) lo = mid;
    else hi = mid;
  }
  const left = points[lo];
  const right = points[hi];
  const span = right.x - left.x;
  if (!(span > 0)) return left.tau;
  const fraction = (step - left.x) / span;
  const lrAtStep = left.y + fraction * (right.y - left.y);
  return left.tau + (step - left.x) * (left.y + lrAtStep) / 2;
}

function xAxisValueForStep(step, rid = runId) {
  if (xAxisMode === "step") return step;
  return learningRateIntegralAt(lossLogByRun.get(rid), step);
}

function tauZeroPlotX() {
  const candidates = [];
  for (const log of lossLogByRun.values()) {
    const tauAtOne = learningRateIntegralAt(log, 1);
    if (Number.isFinite(tauAtOne) && tauAtOne > 0) candidates.push(tauAtOne / 10);
  }
  return candidates.length ? Math.min(...candidates) : Number.EPSILON;
}

function mapPointsToXAxis(rawPoints, logX, logY = logX) {
  const hasNonPositiveY = logY && rawPoints.some(
    (p) => Number.isFinite(p._axisX) && Number.isFinite(p.y) && p.y <= 0,
  );
  const valid = rawPoints.filter(
    (p) => Number.isFinite(p._axisX) && Number.isFinite(p.y) && (!logY || p.y > 0),
  );
  const zeroPlotX =
    xAxisMode === "step"
      ? LOG_ZERO_PLOT_X
      : tauZeroPlotX();
  const mapped = valid
    .filter((p) => !logX || p._axisX >= 0)
    .map((p) => ({
      x: logX && p._axisX === 0 ? zeroPlotX : p._axisX,
      y: p.y,
      _step: p._step,
      _axisX: p._axisX,
      ...(xAxisMode === "tau" ? { _tau: p._axisX } : {}),
    }));
  mapped._logYBlocked = hasNonPositiveY;
  return mapped;
}

function pointsFromSeries(series, rid = runId) {
  const logX = CURVE_LOGLOG_ENABLED && curveScaleMode !== "linear";
  const residual = setupResidualMode && compareRuns;
  const logY = CURVE_LOGLOG_ENABLED && curveScaleMode === "loglog" && !residual;
  const raw = [];
  for (let i = 0; i < series.steps.length; i += 1) {
    const step = series.steps[i];
    const y = series.values[i];
    if (!Number.isFinite(step) || !Number.isFinite(y)) continue;
    raw.push({ _axisX: xAxisValueForStep(step, rid), y, _step: step });
  }
  return mapPointsToXAxis(raw, logX, logY);
}

function lineDataHasPoints(lineData) {
  return !!(lineData?.datasets || []).some((d) => (d.data || []).length);
}

function datasetsHaveBlockedLogY(datasets) {
  return !!(datasets || []).some((dataset) => dataset?.data?._logYBlocked);
}

function updateCurveScaleButtons() {
  document.querySelectorAll(".curve-scale-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.scale === curveScaleMode);
  });
}

function updateXAxisButtons() {
  document.querySelectorAll(".x-axis-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.xAxis === xAxisMode);
  });
}

async function setXAxisMode(mode) {
  if (!["step", "tau"].includes(mode) || mode === xAxisMode) return;
  if (mode === "tau") {
    const ids = new Set([runId]);
    if (compareRuns || compareLossRuns) {
      for (const rid of selectedRunIds()) ids.add(rid);
    }
    await Promise.all([...ids].map((rid) => ensureLossLogForRun(rid)));
    if (![...ids].some((rid) => lossLogByRun.get(rid)?.lr?.length)) return;
  }
  xAxisMode = mode;
  for (const line of referenceLines) {
    line.intercept = null;
    line.mode = null;
  }
  updateXAxisButtons();
  renderLossChart();
  const spec = activeSpec();
  if (spec) renderChart(spec);
  if (fullOverlayOpen && fullOverlayMode === "loss") renderFullLossChart();
}

function setCurveScaleMode(mode) {
  if (!CURVE_LOGLOG_ENABLED) {
    curveScaleMode = "linear";
    return;
  }
  if (!["linear", "loglinear", "loglog"].includes(mode)) return;
  curveScaleMode = mode;
  updateCurveScaleButtons();
  const spec = activeSpec();
  if (spec) renderChart(spec);
  if (fullOverlayOpen && fullOverlayMode === "spec" && spec) renderFullChart(spec);
}

function setCurveOverlayChrome(show) {
  const el = document.getElementById("curveScaleToggleFull");
  if (el) el.hidden = !show || !CURVE_LOGLOG_ENABLED;
  const xAxis = document.getElementById("xAxisToggleFull");
  if (xAxis) xAxis.hidden = !show;
  if (show && CURVE_LOGLOG_ENABLED) updateCurveScaleButtons();
  if (show) updateXAxisButtons();
  updateFullscreenCompareChrome();
}

function buildLineDatasets(spec) {
  if (compareRuns && canCompareRuns()) {
    const residual = setupResidualMode;
    const baselineId = residual ? resolveBaselineRunId() : null;
    let baselinePoints = null;
    if (residual) {
      if (!baselineId) {
        return { empty: true, emptyMessage: "No baseline setup found for residuals" };
      }
      const baseCache = manifestCache.get(baselineId);
      const baseSpec = baseCache?.specById?.get(spec.id);
      if (!baseSpec?.series?.steps?.length) {
        return {
          empty: true,
          emptyMessage: `Baseline (${runLabel(baselineId)}) has no matching series`,
        };
      }
      baselinePoints = pointsFromSeries(baseSpec.series, baselineId);
    }

    const datasets = [];
    const missing = [];
    if (residual && baselinePoints) {
      datasets.push({
        label: `${runLabel(baselineId)} (0)`,
        data: zeroBaselinePoints(baselinePoints),
        borderColor: "rgba(255, 255, 255, 0.55)",
        backgroundColor: "transparent",
        borderWidth: 1.5,
        borderDash: [5, 4],
        pointRadius: 0,
        tension: 0,
        fill: false,
      });
    }

    for (const run of availableRuns) {
      if (!selectedRuns.has(run.run_id)) continue;
      if (residual && run.run_id === baselineId) continue;
      const cache = manifestCache.get(run.run_id);
      const other = cache?.specById?.get(spec.id);
      if (!other?.series?.steps?.length) {
        missing.push(runLabel(run.run_id));
        continue;
      }
      const color = runColor(run.run_id);
      const isCurrent = run.run_id === runId;
      const raw = pointsFromSeries(other.series, run.run_id);
      const data = residual ? residualAgainstBaseline(raw, baselinePoints) : raw;
      if (residual && !data.length) {
        missing.push(runLabel(run.run_id));
        continue;
      }
      datasets.push({
        label: residual ? `Δ ${runLabel(run.run_id)}` : runLabel(run.run_id),
        data,
        borderColor: color,
        backgroundColor: `${color}22`,
        borderWidth: isCurrent ? 2.5 : 1.75,
        borderDash: isCurrent ? [] : [6, 3],
        pointRadius: 0,
        tension: 0.2,
        fill: false,
      });
    }
    if (!datasets.length || (residual && datasets.length <= 1 && missing.length)) {
      return {
        empty: true,
        emptyMessage: runsLoading
          ? "Loading setup data…"
          : missing.length
            ? `No matching series for: ${missing.join(", ")}`
            : "Select at least one setup above",
      };
    }
    if (residual && datasets.length === 1) {
      return {
        empty: true,
        emptyMessage: "Select another setup besides baseline to plot residuals",
      };
    }
    return {
      legend: true,
      datasets,
      residual,
      missingNote: missing.length ? `missing in ${missing.join(", ")}` : "",
    };
  }
  if (compareLayers && canCompareLayers(spec)) {
    const members = familyMembers(spec)
      .filter((m) => selectedLayers.has(m.layer))
      .sort((a, b) => a.layer - b.layer);
    if (!members.length) return { empty: true, emptyMessage: "Select layers to compare above" };
    return {
      legend: true,
      datasets: members.map((m) => {
        const color = LAYER_COLORS[m.layer % LAYER_COLORS.length];
        return {
          label: `L${m.layer}`,
          data: pointsFromSeries(m.series, runId),
          borderColor: color,
          backgroundColor: `${color}22`,
          borderWidth: m.layer === activeLayer ? 2.5 : 1.5,
          pointRadius: 0,
          tension: 0.2,
          fill: false,
        };
      }),
    };
  }
  if (spec.series?.steps?.length) {
    const color = SOURCE_COLORS[spec.source_kind] || "#7eb8c9";
    return {
      legend: false,
      datasets: [{
        label: displaySpecLabel(spec),
        data: pointsFromSeries(spec.series, runId),
        borderColor: color,
        backgroundColor: `${color}22`,
        borderWidth: 2.75,
        pointRadius: 2,
        tension: 0.25,
        fill: true,
      }],
    };
  }
  return null;
}

function chartTitleFor(spec) {
  const label = displaySpecLabel(spec);
  if (compareRuns && canCompareRuns()) {
    return setupResidualMode
      ? `${label} · residuals vs baseline`
      : `${label} · setup compare`;
  }
  if (compareLayers && canCompareLayers(spec)) return `${label} · layer compare`;
  return label;
}

function chartInfoFor(spec, lineData = null) {
  if (compareRuns && canCompareRuns()) {
    if (setupResidualMode) {
      const bid = resolveBaselineRunId();
      let base = `baseline: ${runLabel(bid)} = 0`;
      if (lineData?.missingNote) base = `${base} · ${lineData.missingNote}`;
      return base;
    }
    const labels = availableRuns
      .filter((r) => selectedRuns.has(r.run_id))
      .map((r) => runLabel(r.run_id));
    let base = labels.join(" vs ") || "no setup";
    if (lineData?.missingNote) base = `${base} · ${lineData.missingNote}`;
    return base;
  }
  if (compareLayers && canCompareLayers(spec)) {
    return (
      familyMembers(spec)
        .filter((m) => selectedLayers.has(m.layer))
        .sort((a, b) => a.layer - b.layer)
        .map((m) => `L${m.layer}`)
        .join(", ") || "no layers selected"
    );
  }
  return "";
}

function renderSpecDefinition(spec) {
  const el = document.getElementById("chartDef");
  if (!el) return;
  if (!spec) {
    definitionResizeObserver?.disconnect();
    el.hidden = true;
    el.innerHTML = "";
    return;
  }

  const direct = typeof buildSpecDirectFormula === "function"
    ? buildSpecDirectFormula(spec)
    : null;
  if (!direct?.tex) {
    definitionResizeObserver?.disconnect();
    el.hidden = true;
    el.innerHTML = "";
    return;
  }

  let mathHtml;
  if (window.katex) {
    try {
      mathHtml = katex.renderToString(direct.tex, {
        throwOnError: true,
        displayMode: true,
        fleqn: true,
        output: "htmlAndMathml",
      });
    } catch (_) {
      mathHtml = `<code>${escapeHtml(direct.tex)}</code>`;
    }
  } else {
    mathHtml = `<code>${escapeHtml(direct.tex)}</code>`;
  }
  const href = typeof referenceAnchorForSpec === "function"
    ? referenceAnchorForSpec(spec)
    : "reference.html";
  const desc =
    direct.description ||
    (typeof buildSpecPlainDescription === "function" ? buildSpecPlainDescription(spec) : "");
  const formulaLineCount = direct.tex.split("\\\\[2pt]").length;
  el.classList.toggle("chart-def-dense", formulaLineCount >= 5);
  el.innerHTML = [
    `<div class="chart-def-row"><span class="chart-def-label">Definition</span>` +
      `<a class="chart-def-link" href="${href}" target="_blank" rel="noopener">All formulas</a></div>`,
    desc ? `<p class="chart-def-desc">${escapeHtml(desc)}</p>` : "",
    `<div class="chart-def-math">` +
      `<div class="chart-def-eq" role="region" tabindex="0" ` +
        `aria-label="Exact mathematical definition" title="${escapeHtml(direct.title)}">` +
        `${mathHtml}</div>` +
      `<p class="chart-def-scroll-hint" hidden>Scroll horizontally to view the complete formula</p>` +
    `</div>`,
  ].filter(Boolean).join("");
  el.hidden = false;

  const equation = el.querySelector(".chart-def-eq");
  const hint = el.querySelector(".chart-def-scroll-hint");
  const updateOverflowState = () => {
    if (!equation?.isConnected) return;
    const overflowing = equation.scrollWidth > equation.clientWidth + 1;
    equation.classList.toggle("is-overflowing", overflowing);
    if (hint) hint.hidden = !overflowing;
  };
  definitionResizeObserver?.disconnect();
  if (equation && typeof ResizeObserver !== "undefined") {
    definitionResizeObserver = new ResizeObserver(updateOverflowState);
    definitionResizeObserver.observe(equation);
  }
  requestAnimationFrame(updateOverflowState);
}

function renderChart(spec) {
  const titleEl = document.getElementById("chartTitle");
  const infoEl = document.getElementById("chartInfo");
  const canvas = document.getElementById("curveChart");
  const image = document.getElementById("curveImage");

  canvas.hidden = false;
  image.hidden = true;
  image.classList.remove("visible");
  image.removeAttribute("src");
  updateCompareToggle(spec);
  renderSpecDefinition(spec);

  if (!spec) {
    titleEl.textContent = "Select an observable";
    infoEl.textContent = "";
    setChartChrome(false);
    destroyChart();
    return;
  }

  destroyChart();

  const lineData = buildLineDatasets(spec);
  titleEl.textContent = chartTitleFor(spec);
  infoEl.textContent = chartInfoFor(spec, lineData);

  if (lineData?.empty) {
    setChartChrome(false);
    infoEl.textContent = lineData.emptyMessage || "Select layers to compare above";
    if (fullOverlayOpen && fullOverlayMode === "spec") renderFullChart(spec);
    return;
  }

  if (lineData) {
    if (datasetsHaveBlockedLogY(lineData.datasets)) {
      setChartChrome(true);
      infoEl.textContent = LOG_SCALE_NON_POSITIVE_MESSAGE;
      if (fullOverlayOpen && fullOverlayMode === "spec") renderFullChart(spec);
      return;
    }
    if (!lineDataHasPoints(lineData)) {
      setChartChrome(true);
      infoEl.textContent =
        curveScaleMode === "loglog"
          ? "No positive step/value points for log–log view."
          : "No curve data yet.";
      if (fullOverlayOpen && fullOverlayMode === "spec") renderFullChart(spec);
      return;
    }
    chart = new Chart(canvas, {
      type: "line",
      data: { datasets: lineData.datasets },
      options: chartCommonOptions({
        legend: lineData.legend,
        scaleMode: curveScaleMode,
        residual: !!lineData.residual,
      }),
    });
    attachDoubleClickReset(canvas);
    setChartChrome(true);
    if (fullOverlayOpen && fullOverlayMode === "spec") renderFullChart(spec);
    return;
  }

  setChartChrome(false);

  if (spec.curve_png) {
    canvas.hidden = true;
    image.hidden = false;
    image.onload = () => image.classList.add("visible");
    image.onerror = () => {
      infoEl.textContent = "PNG curve not found yet — training may still be running.";
    };
    image.src = `data/${runId}/${spec.curve_png}?t=${Date.now()}`;
    return;
  }

  infoEl.textContent = "No curve data yet.";
}

/* ---------- Loss chart ---------- */

function parseLossCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return null;

  const parseRow = (line) => {
    const columns = [];
    let value = "";
    let quoted = false;
    for (let i = 0; i < line.length; i += 1) {
      const char = line[i];
      if (char === '"') {
        if (quoted && line[i + 1] === '"') {
          value += '"';
          i += 1;
        } else {
          quoted = !quoted;
        }
      } else if (char === "," && !quoted) {
        columns.push(value);
        value = "";
      } else {
        value += char;
      }
    }
    columns.push(value);
    return columns;
  };

  const headers = parseRow(lines[0])
    .map((h, index) => (index === 0 ? h.replace(/^\uFEFF/, "") : h).trim());
  const col = (name) => headers.indexOf(name);
  const iterIdx = col("iter");
  const trainIdx = col("train_loss");
  const valIdx = col("val_loss");
  const lrIdx = col("lr");
  if (iterIdx < 0 || trainIdx < 0 || valIdx < 0) return null;

  const trainByStep = new Map();
  const valByStep = new Map();
  const lrByStep = new Map();
  const parseNumber = (raw) => {
    if (raw == null || String(raw).trim() === "") return NaN;
    return Number(raw);
  };
  for (let i = 1; i < lines.length; i += 1) {
    if (!lines[i].trim()) continue;
    const cols = parseRow(lines[i]);
    const x = parseNumber(cols[iterIdx]);
    const yt = parseNumber(cols[trainIdx]);
    const yv = parseNumber(cols[valIdx]);
    const lr = lrIdx >= 0 ? parseNumber(cols[lrIdx]) : NaN;
    if (Number.isFinite(x) && Number.isFinite(yt)) trainByStep.set(x, yt);
    if (Number.isFinite(x) && Number.isFinite(yv)) valByStep.set(x, yv);
    if (Number.isFinite(x) && Number.isFinite(lr) && lr >= 0) lrByStep.set(x, lr);
  }
  const toPoints = (values) =>
    [...values.entries()]
      .sort(([a], [b]) => a - b)
      .map(([x, y]) => ({ x, y }));
  const train = toPoints(trainByStep);
  const val = toPoints(valByStep);
  const lr = toPoints(lrByStep);
  let tau = 0;
  for (let i = 0; i < lr.length; i += 1) {
    if (i > 0) {
      const prev = lr[i - 1];
      tau += (lr[i].x - prev.x) * (prev.y + lr[i].y) / 2;
    }
    lr[i].tau = tau;
  }
  if (!train.length && !val.length) return null;
  return { train, val, lr };
}

async function loadLossLog() {
  const text = await fetchText(`data/${runId}/eval_loss_log.csv`);
  if (!text) return { error: true };
  const parsed = parseLossCsv(text);
  if (!parsed) return { error: true };
  return parsed;
}

function lossLogsForChart() {
  if (compareLossRuns && canCompareRuns()) {
    const want = new Set(selectedRunIds());
    return availableRuns
      .filter((r) => want.has(r.run_id))
      .map((r) => ({ run_id: r.run_id, label: runLabel(r.run_id), log: lossLogByRun.get(r.run_id) }))
      .filter((x) => x.log && !x.log.error);
  }
  if (!lossLog || lossLog.error) return [];
  return [{ run_id: runId, label: runLabel(runId), log: lossLog }];
}

function lossDataBounds() {
  const logs = lossLogsForChart();
  const xs = [];
  for (const item of logs) {
    for (const p of [...item.log.train, ...item.log.val]) xs.push(p.x);
  }
  const bounds = finiteMinMax(xs);
  return bounds || { min: 0, max: 0 };
}

/** Sorted unique true steps available in the current loss data. */
function lossAvailableSteps() {
  const logs = lossLogsForChart();
  const xs = new Set();
  for (const item of logs) {
    for (const p of [...item.log.train, ...item.log.val]) {
      if (Number.isFinite(p.x)) xs.add(p.x);
    }
  }
  return [...xs].sort((a, b) => a - b);
}

/** Smallest data step ≥ value (clamp to last step if past the end). */
function ceilToAvailableStep(value, steps) {
  if (!Number.isFinite(value) || !steps.length) return null;
  for (const s of steps) {
    if (s >= value) return s;
  }
  return steps[steps.length - 1];
}

/** Largest data step <= value (clamp to the first step if before the start). */
function floorToAvailableStep(value, steps) {
  if (!Number.isFinite(value) || !steps.length) return null;
  for (let i = steps.length - 1; i >= 0; i -= 1) {
    if (steps[i] <= value) return steps[i];
  }
  return steps[0];
}

function syncLossRangeInputs() {
  const bounds = lossDataBounds();
  if (lossLog?.error) return;
  for (const [minId, maxId] of [
    ["lossStepMin", "lossStepMax"],
    ["lossStepMinFull", "lossStepMaxFull"],
  ]) {
    const minEl = document.getElementById(minId);
    const maxEl = document.getElementById(maxId);
    if (!minEl || !maxEl) continue;
    minEl.placeholder = String(bounds.min);
    maxEl.placeholder = String(bounds.max);
    minEl.value = lossStepMin == null ? "" : String(lossStepMin);
    maxEl.value = lossStepMax == null ? "" : String(lossStepMax);
  }
}

function applyLossStepRange() {
  const minEl = document.getElementById("lossStepMin");
  const maxEl = document.getElementById("lossStepMax");
  const steps = lossAvailableSteps();
  let min = minEl.value === "" ? null : Number(minEl.value);
  let max = maxEl.value === "" ? null : Number(maxEl.value);
  if (min != null && !Number.isFinite(min)) min = null;
  if (max != null && !Number.isFinite(max)) max = null;
  // Snap the lower bound up and the upper bound down to actual data steps.
  if (min != null) min = ceilToAvailableStep(min, steps);
  if (max != null) max = floorToAvailableStep(max, steps);
  if (min != null && max != null && min > max) [min, max] = [max, min];
  lossStepMin = min;
  lossStepMax = max;
  syncLossRangeInputs();
  renderLossChart();
  if (fullOverlayOpen && fullOverlayMode === "loss") renderFullLossChart();
}

function resetLossStepRange() {
  lossStepMin = null;
  lossStepMax = null;
  syncLossRangeInputs();
  renderLossChart();
  if (fullOverlayOpen && fullOverlayMode === "loss") renderFullLossChart();
}

function filterLossPoints(points, rid = runId) {
  const logX = lossScaleMode !== "linear";
  const residual = setupResidualMode && compareLossRuns;
  const logY = lossScaleMode === "loglog" && !residual;
  const raw = points
    .filter((p) => {
      if (lossStepMin != null && p.x < lossStepMin) return false;
      if (lossStepMax != null && p.x > lossStepMax) return false;
      return Number.isFinite(p.x) && Number.isFinite(p.y);
    })
    .map((p) => ({ _axisX: xAxisValueForStep(p.x, rid), y: p.y, _step: p.x }));
  return mapPointsToXAxis(raw, logX, logY);
}

function lossChartScaleOptions() {
  const axis = "rgba(255, 255, 255, 0.78)";
  const grid = "rgba(255, 255, 255, 0.12)";
  const residual = setupResidualMode && compareLossRuns;
  const logX = lossScaleMode !== "linear";
  const logY = !residual && lossScaleMode === "loglog";
  return {
    x: {
      type: logX ? "logarithmic" : "linear",
      title: {
        display: true,
        text: logX ? `${xAxisTitle()} (log)` : xAxisTitle(),
        color: axis,
      },
      ticks: logX ? logXAxisTickConfig(axis) : { color: axis },
      ...(logX ? { afterBuildTicks: afterBuildLogXTicks } : {}),
      grid: { color: grid },
    },
    y: {
      type: logY ? "logarithmic" : "linear",
      title: {
        display: true,
        text: residual ? "Δ Loss vs baseline" : logY ? "Loss (log)" : "Loss",
        color: axis,
      },
      ticks: { color: axis },
      grid: { color: grid },
    },
  };
}

function lossChartOptions({ enablePan = true, onClick = null } = {}) {
  const residual = setupResidualMode && compareLossRuns;
  const opts = chartCommonOptions({
    legend: true,
    enablePan,
    onClick,
    scaleMode: lossScaleMode,
    yTitle: residual ? "Δ Loss vs baseline" : "Loss",
    residual,
  });
  opts.scales = lossChartScaleOptions();
  return opts;
}

function destroyLossChart() {
  if (lossChart) {
    lossChart.destroy();
    lossChart = null;
  }
}

function lossInfoText() {
  const logs = lossLogsForChart();
  if (compareLossRuns && canCompareRuns()) {
    const want = new Set(selectedRunIds());
    const missing = availableRuns
      .filter((r) => want.has(r.run_id))
      .filter((r) => {
        const log = lossLogByRun.get(r.run_id);
        return !log || log.error;
      })
      .map((r) => runLabel(r.run_id));
    if (!logs.length) {
      return missing.length
        ? `No loss data (${missing.join(", ")})`
        : "No loss data";
    }
    if (missing.length) return `Missing: ${missing.join(", ")}`;
    return logs.map((x) => x.label).join(" vs ");
  }
  return "";
}

function buildLossDatasets() {
  const logs = lossLogsForChart();
  if (!logs.length) return null;
  const datasets = [];
  const residual = setupResidualMode && compareLossRuns && canCompareRuns();

  if (compareLossRuns && canCompareRuns()) {
    let baselineItem = null;
    if (residual) {
      const bid = resolveBaselineRunId();
      baselineItem =
        logs.find((x) => x.run_id === bid) ||
        (() => {
          const log = lossLogByRun.get(bid);
          return log && !log.error
            ? { run_id: bid, label: runLabel(bid), log }
            : null;
        })();
      if (!baselineItem) return null;
    }

    if (residual && baselineItem) {
      const modes = [];
      if (lossViewMode === "both" || lossViewMode === "train") modes.push(["train", "train"]);
      if (lossViewMode === "both" || lossViewMode === "val") modes.push(["val", "val"]);
      for (const [key, tag] of modes) {
        const basePts = filterLossPoints(baselineItem.log[key], baselineItem.run_id);
        datasets.push({
          label: `${baselineItem.label} · ${tag} (0)`,
          data: zeroBaselinePoints(basePts),
          borderColor: "rgba(255, 255, 255, 0.55)",
          backgroundColor: "transparent",
          borderWidth: 1.5,
          borderDash: [5, 4],
          pointRadius: 0,
          tension: 0,
          fill: false,
        });
      }
    }

    for (const item of logs) {
      if (residual && item.run_id === baselineItem.run_id) continue;
      const color = runColor(item.run_id);
      const isCurrent = item.run_id === runId;
      if (lossViewMode === "both" || lossViewMode === "train") {
        const raw = filterLossPoints(item.log.train, item.run_id);
        const data = residual
          ? residualAgainstBaseline(
              raw,
              filterLossPoints(baselineItem.log.train, baselineItem.run_id),
            )
          : raw;
        if (data.length) {
          datasets.push({
            label: residual ? `Δ ${item.label} · train` : `${item.label} · train`,
            data,
            borderColor: color,
            backgroundColor: `${color}22`,
            borderWidth: isCurrent ? 2.5 : 1.75,
            borderDash: [],
            pointRadius: 0,
            tension: 0.2,
            fill: false,
          });
        }
      }
      if (lossViewMode === "both" || lossViewMode === "val") {
        const raw = filterLossPoints(item.log.val, item.run_id);
        const data = residual
          ? residualAgainstBaseline(
              raw,
              filterLossPoints(baselineItem.log.val, baselineItem.run_id),
            )
          : raw;
        if (data.length) {
          datasets.push({
            label: residual ? `Δ ${item.label} · val` : `${item.label} · val`,
            data,
            borderColor: color,
            backgroundColor: `${color}22`,
            borderWidth: isCurrent ? 2.5 : 1.75,
            borderDash: [6, 3],
            pointRadius: 0,
            tension: 0.2,
            fill: false,
          });
        }
      }
    }
    if (!datasets.length) return null;
    if (!datasets.some((d) => (d.data || []).length)) return null;
    return datasets;
  }

  const item = logs[0];
  if (lossViewMode === "both" || lossViewMode === "train") {
    datasets.push({
      label: "Train loss",
      data: filterLossPoints(item.log.train, item.run_id),
      borderColor: "#9fd4e4",
      backgroundColor: "rgba(159, 212, 228, 0.2)",
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.2,
      fill: false,
    });
  }
  if (lossViewMode === "both" || lossViewMode === "val") {
    datasets.push({
      label: "Val loss",
      data: filterLossPoints(item.log.val, item.run_id),
      borderColor: "#e8c888",
      backgroundColor: "rgba(232, 200, 136, 0.2)",
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.2,
      fill: false,
    });
  }
  if (!datasets.length) return null;
  if (!datasets.some((d) => (d.data || []).length)) return null;
  return datasets;
}

function createLossChart(
  canvas,
  { enablePan = true, onClick = null, datasets = null } = {},
) {
  const chartDatasets = datasets || buildLossDatasets();
  if (!chartDatasets || datasetsHaveBlockedLogY(chartDatasets)) return null;
  return new Chart(canvas, {
    type: "line",
    data: { datasets: chartDatasets },
    options: lossChartOptions({ enablePan, onClick }),
  });
}

function updateLossViewButtons() {
  document.querySelectorAll(".loss-view-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === lossViewMode);
  });
  document.querySelectorAll(".loss-scale-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.scale === lossScaleMode);
  });
  updateXAxisButtons();
}

function setLossOverlayChrome(show) {
  const el = document.getElementById("lossViewToggleFull");
  if (el) el.hidden = !show;
  const scale = document.getElementById("lossScaleToggleFull");
  if (scale) scale.hidden = !show;
  const xAxis = document.getElementById("xAxisToggleFull");
  if (xAxis) xAxis.hidden = !show;
  const range = document.getElementById("lossRangeFull");
  if (range) range.hidden = !show;
  updateFullscreenCompareChrome();
}

function setLossViewMode(mode) {
  if (!["both", "train", "val"].includes(mode)) return;
  lossViewMode = mode;
  updateLossViewButtons();
  renderLossChart();
  if (fullOverlayOpen && fullOverlayMode === "loss") renderFullLossChart();
}

function setLossScaleMode(mode) {
  if (!["linear", "loglinear", "loglog"].includes(mode)) return;
  lossScaleMode = mode;
  updateLossViewButtons();
  renderLossChart();
  if (fullOverlayOpen && fullOverlayMode === "loss") renderFullLossChart();
}

function renderLossChart() {
  const section = document.getElementById("lossSection");
  const infoEl = document.getElementById("lossInfo");
  const canvas = document.getElementById("lossChart");
  if (!section || !infoEl || !canvas) return;

  destroyLossChart();
  updateLossViewButtons();
  updateLossCompareToggle();

  if (lossLog?.error) {
    section.hidden = false;
    infoEl.textContent = "eval_loss_log.csv not found";
    return;
  }

  if (!lossLog) {
    section.hidden = true;
    return;
  }

  section.hidden = false;
  infoEl.textContent = lossInfoText();
  const datasets = buildLossDatasets();
  if (datasetsHaveBlockedLogY(datasets)) {
    infoEl.textContent = LOG_SCALE_NON_POSITIVE_MESSAGE;
    return;
  }
  lossChart = createLossChart(canvas, { datasets });
  if (!lossChart) {
    infoEl.textContent =
      lossScaleMode === "loglog"
        ? "No positive step/loss points for log–log view."
        : lossInfoText() || "No loss data to display.";
    return;
  }
  canvas.ondblclick = (evt) => {
    evt.preventDefault();
    renderLossChart();
  };
}

function destroyFullChart() {
  if (!fullChart) return;
  try {
    detachReferenceLineDragHandlers();
    if (typeof detachFullscreenNoteHandlers === "function") {
      detachFullscreenNoteHandlers(fullChart);
    }
    fullChart.destroy();
  } catch (_) {
    /* ignore */
  }
  fullChart = null;
}

function fullscreenNoteClickHandler(evt, _elements, chartInstance) {
  if (typeof handleFullscreenChartClick === "function") {
    handleFullscreenChartClick(evt, chartInstance || fullChart);
  }
}

function bindFullscreenCanvasInteractions(canvas) {
  // Double-click only resets zoom in-place (never destroys the chart).
  canvas.ondblclick = (evt) => {
    evt.preventDefault();
    evt.stopPropagation();
    if (typeof cancelPendingNoteClick === "function") cancelPendingNoteClick();
    safeResetFullChartZoom();
  };
  attachReferenceLineDragHandlers(canvas);
  if (typeof attachFullscreenNoteHandlers === "function") {
    attachFullscreenNoteHandlers(fullChart);
  }
}

function renderFullLossChart() {
  const canvas = document.getElementById("curveChartFull");
  const titleEl = document.getElementById("fullChartTitle");
  const infoEl = document.getElementById("fullChartInfo");
  updateLossViewButtons();
  const datasets = buildLossDatasets();
  titleEl.textContent = "Training Loss";
  infoEl.textContent = lossInfoText();
  if (!datasets) {
    infoEl.textContent =
      lossScaleMode === "loglog"
        ? "No positive step/loss points for log–log view."
        : "No loss data to display.";
    destroyFullChart();
    return;
  }
  if (datasetsHaveBlockedLogY(datasets)) {
    infoEl.textContent = LOG_SCALE_NON_POSITIVE_MESSAGE;
    destroyFullChart();
    return;
  }
  destroyFullChart();
  fullChart = createLossChart(canvas, {
    enablePan: false,
    onClick: fullscreenNoteClickHandler,
    datasets,
  });
  if (fullChart) {
    bindFullscreenCanvasInteractions(canvas);
    applyReferenceLinesToFullChart();
  }
}

function renderFullChart(spec) {
  const canvas = document.getElementById("curveChartFull");
  const titleEl = document.getElementById("fullChartTitle");
  const infoEl = document.getElementById("fullChartInfo");
  updateCurveScaleButtons();
  const lineData = buildLineDatasets(spec);
  titleEl.textContent = chartTitleFor(spec);
  infoEl.textContent = chartInfoFor(spec, lineData);
  if (!lineData || lineData.empty) {
    infoEl.textContent = lineData?.empty
      ? (lineData.emptyMessage || "Select layers to compare above")
      : "No curve data yet.";
    // Never leave a stale chart visible after the selection becomes empty.
    destroyFullChart();
    return;
  }
  if (datasetsHaveBlockedLogY(lineData.datasets)) {
    infoEl.textContent = LOG_SCALE_NON_POSITIVE_MESSAGE;
    destroyFullChart();
    return;
  }
  if (!lineDataHasPoints(lineData)) {
    infoEl.textContent =
      curveScaleMode === "loglog"
        ? "No positive step/value points for log–log view."
        : "No curve data yet.";
    destroyFullChart();
    return;
  }
  destroyFullChart();
  // Clear Chart.js-mutated canvas attrs that can leave a blank surface after destroy.
  canvas.removeAttribute("width");
  canvas.removeAttribute("height");
  canvas.style.width = "";
  canvas.style.height = "";
  fullChart = new Chart(canvas, {
    type: "line",
    data: { datasets: lineData.datasets },
    options: chartCommonOptions({
      legend: lineData.legend,
      onClick: fullscreenNoteClickHandler,
      enablePan: false,
      scaleMode: curveScaleMode,
      residual: !!lineData.residual,
    }),
  });
  bindFullscreenCanvasInteractions(canvas);
  applyReferenceLinesToFullChart();
}

function openLossFullscreen() {
  if (!lossLog || lossLog.error || !buildLossDatasets()) return;
  fullscreenCompareSnapshot = snapshotCompareState();
  fullOverlayMode = "loss";
  fullOverlayOpen = true;
  resetReferenceLines();
  document.getElementById("referenceLineControls").hidden = false;
  setCurveOverlayChrome(false);
  setLossOverlayChrome(true);
  const overlay = document.getElementById("chartOverlay");
  overlay.hidden = false;
  overlay.classList.add("visible");
  overlay.setAttribute("aria-hidden", "false");
  renderFullLossChart();
  if (typeof refreshNotes === "function") refreshNotes();
}

function openFullscreen() {
  const spec = activeSpec();
  if (!spec) return;
  fullscreenCompareSnapshot = snapshotCompareState();
  fullOverlayMode = "spec";
  fullOverlayOpen = true;
  resetReferenceLines();
  document.getElementById("referenceLineControls").hidden = false;
  setLossOverlayChrome(false);
  setCurveOverlayChrome(true);
  const overlay = document.getElementById("chartOverlay");
  overlay.hidden = false;
  overlay.classList.add("visible");
  overlay.setAttribute("aria-hidden", "false");
  renderFullChart(spec);
  if (typeof refreshNotes === "function") refreshNotes();
}

function closeFullscreen() {
  if (typeof cancelPendingNoteClick === "function") cancelPendingNoteClick();
  if (typeof closeNoteModal === "function") closeNoteModal();
  if (typeof setNotesRailExpanded === "function") setNotesRailExpanded(false);
  fullOverlayOpen = false;
  fullOverlayMode = null;
  resetReferenceLines();
  document.getElementById("referenceLineControls").hidden = true;
  setLossOverlayChrome(false);
  setCurveOverlayChrome(false);
  // Fullscreen compare edits must not leak back to the main page.
  const snap = fullscreenCompareSnapshot;
  fullscreenCompareSnapshot = null;
  restoreCompareState(snap);
  updateFullscreenCompareChrome();
  const overlay = document.getElementById("chartOverlay");
  overlay.classList.remove("visible");
  overlay.setAttribute("aria-hidden", "true");
  overlay.hidden = true;
  destroyFullChart();
}

function destroyChart() {
  if (chart) {
    chart.destroy();
    chart = null;
  }
}

function clearSelection() {
  activeModuleId = null;
  activeSpecId = null;
  compareLayers = false;
  compareRuns = false;
  selectedLayers = new Set();
  const input = document.getElementById("compareLayers");
  if (input) input.checked = false;
  const runInput = document.getElementById("compareRuns");
  if (runInput) runInput.checked = false;
  const layerWrap = document.getElementById("compareToggleWrap");
  if (layerWrap) layerWrap.hidden = true;
  const runWrap = document.getElementById("compareRunsWrap");
  if (runWrap) runWrap.hidden = true;
  const layerPick = document.getElementById("layerPickWrap");
  if (layerPick) layerPick.hidden = true;
  const runPick = document.getElementById("runPickWrap");
  if (runPick) runPick.hidden = true;
  setRunPickStatus("");
  setChartChrome(false);
  closeFullscreen();
  document.getElementById("detailEmpty").style.display = "grid";
  document.getElementById("detailContent").classList.remove("visible");
  destroyChart();
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

(async function startApp() {
  if (typeof MaintenanceGate !== "undefined") {
    const ok = await MaintenanceGate.wire();
    if (!ok) return;
  }
  await boot();
})();
