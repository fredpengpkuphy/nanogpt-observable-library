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

let manifest = null;
let runId = null;
let availableRuns = [];
let manifestCache = new Map();
let lossLogByRun = new Map();
let compareRuns = false;
let compareLossRuns = false;
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
/** @type {"linear" | "loglog"} */
let lossScaleMode = "linear";
/** @type {"linear" | "loglog"} — observables temporarily locked to linear. */
let curveScaleMode = "linear";
const CURVE_LOGLOG_ENABLED = false;
let lossStepMin = null;
let lossStepMax = null;
let fullChart = null;
let fullOverlayOpen = false;
let fullOverlayMode = null;

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
    specById = new Map(manifest.specs.map((s) => [s.id, s]));
    specsByModule = groupSpecsByModule(manifest.specs);
    specsByFamily = groupSpecsByFamily(manifest.specs);
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
  document.getElementById("pickAllRuns")?.addEventListener("click", async () => {
    selectedRuns = new Set(availableRuns.map((r) => r.run_id));
    renderRunPicker();
    await ensureSelectedRunManifests();
    const spec = activeSpec();
    if (spec) renderChart(spec);
  });
  document.getElementById("pickCurrentRun")?.addEventListener("click", () => {
    selectedRuns = new Set([runId]);
    renderRunPicker();
    const spec = activeSpec();
    if (spec) renderChart(spec);
  });
  document.getElementById("resetZoomBtn").addEventListener("click", resetChartZoom);
  document.getElementById("expandBtn").addEventListener("click", openFullscreen);
  document.getElementById("fullCloseBtn").addEventListener("click", closeFullscreen);
  document.getElementById("fullResetZoomBtn").addEventListener("click", () => {
    safeResetFullChartZoom();
  });
  document.getElementById("chartOverlay").addEventListener("click", (e) => {
    if (e.target.id === "chartOverlay") closeFullscreen();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || !fullOverlayOpen) return;
    const noteModal = document.getElementById("noteModal");
    if (noteModal && !noteModal.hidden) return;
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
    const map = new Map(m.specs.map((s) => [s.id, s]));
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
  }
  const spec = activeSpec();
  updateCompareToggle(spec);
  if (spec) renderChart(spec);
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

async function setCompareLossRuns(on) {
  compareLossRuns = on;
  if (on) {
    await Promise.all(selectedRunIds().map((rid) => ensureLossLogForRun(rid)));
  }
  updateLossCompareToggle();
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
    return;
  }
  input.checked = compareLossRuns;
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
  const m = manifest.model;
  const label = await resolveRunLabel();
  document.getElementById("runSubtitle").textContent =
    `${label} · ${m.name} · ${m.n_layer} layers · ${m.n_embd} dim · ${manifest.n_specs} observables`;
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
      chip.textContent = spec.label;
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
      await ensureSelectedRunManifests();
      const spec = activeSpec();
      if (spec) renderChart(spec);
      if (compareLossRuns) {
        await setCompareLossRuns(true);
      }
    });
    list.appendChild(label);
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

function chartScaleOptions({
  scaleMode = "linear",
  xTitle = "Step",
  yTitle = "Value",
} = {}) {
  const axis = "rgba(255, 255, 255, 0.78)";
  const grid = "rgba(255, 255, 255, 0.12)";
  const log = scaleMode === "loglog";
  return {
    x: {
      type: log ? "logarithmic" : "linear",
      title: { display: true, text: log ? `${xTitle} (log)` : xTitle, color: axis },
      ticks: {
        color: axis,
        // Points are stored at step+1 so step 0 is plottable; labels show the true step.
        ...(log
          ? {
              callback(value) {
                const step = Number(value) - 1;
                if (!Number.isFinite(step)) return "";
                return Math.abs(step - Math.round(step)) < 1e-9
                  ? String(Math.round(step))
                  : String(step);
              },
            }
          : {}),
      },
      grid: { color: grid },
    },
    y: {
      type: log ? "logarithmic" : "linear",
      title: { display: true, text: log ? `${yTitle} (log)` : yTitle, color: axis },
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
    },
    pan: {
      // Fullscreen charts disable pan so Hammer does not interfere with note clicks.
      enabled: enablePan,
      mode: "xy",
      modifierKey: enablePan ? "alt" : null,
      threshold: 10,
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
} = {}) {
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
            if (raw && Number.isFinite(raw._step)) return `step ${raw._step}`;
            const x = items[0].parsed.x;
            if (scaleMode === "loglog" && Number.isFinite(x)) {
              return `step ${Math.round(x - 1)}`;
            }
            return `step ${x}`;
          },
        },
      },
      zoom: zoomPluginOptions({ enablePan }),
    },
    scales: chartScaleOptions({ scaleMode, yTitle }),
  };
  if (typeof onClick === "function") opts.onClick = onClick;
  return opts;
}

/** Restore axis range from data without destroying the Chart instance. */
function fitChartScalesToData(chart) {
  if (!chart?.data?.datasets?.length) return false;
  const xs = [];
  const ys = [];
  for (const ds of chart.data.datasets) {
    for (const p of ds.data || []) {
      if (p && Number.isFinite(p.x)) xs.push(p.x);
      if (p && Number.isFinite(p.y)) ys.push(p.y);
    }
  }
  if (!xs.length || !ys.length) return false;
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
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
  }
}

function setChartChrome(hasSeriesChart) {
  document.getElementById("resetZoomBtn").hidden = !hasSeriesChart;
  document.getElementById("chartHint").hidden = !hasSeriesChart;
  document.getElementById("expandBtn").hidden = !hasSeriesChart;
  const scale = document.getElementById("curveScaleToggle");
  if (scale) scale.hidden = !hasSeriesChart || !CURVE_LOGLOG_ENABLED;
  updateCurveScaleButtons();
}

function pointsFromSeries(series) {
  const log = CURVE_LOGLOG_ENABLED && curveScaleMode === "loglog";
  const pts = [];
  for (let i = 0; i < series.steps.length; i += 1) {
    const step = series.steps[i];
    const y = series.values[i];
    if (!Number.isFinite(step) || !Number.isFinite(y)) continue;
    // Log-y still needs y > 0; x uses step+1 so step 0 appears as label "0".
    if (log && !(y > 0)) continue;
    pts.push({
      x: log ? step + 1 : step,
      y,
      _step: step,
    });
  }
  return pts;
}

function lineDataHasPoints(lineData) {
  return !!(lineData?.datasets || []).some((d) => (d.data || []).length);
}

function updateCurveScaleButtons() {
  document.querySelectorAll(".curve-scale-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.scale === curveScaleMode);
  });
}

function setCurveScaleMode(mode) {
  if (!CURVE_LOGLOG_ENABLED) {
    curveScaleMode = "linear";
    return;
  }
  if (!["linear", "loglog"].includes(mode)) return;
  curveScaleMode = mode;
  updateCurveScaleButtons();
  const spec = activeSpec();
  if (spec) renderChart(spec);
  if (fullOverlayOpen && fullOverlayMode === "spec" && spec) renderFullChart(spec);
}

function setCurveOverlayChrome(show) {
  const el = document.getElementById("curveScaleToggleFull");
  if (el) el.hidden = !show || !CURVE_LOGLOG_ENABLED;
  if (show && CURVE_LOGLOG_ENABLED) updateCurveScaleButtons();
}

function buildLineDatasets(spec) {
  if (compareRuns && canCompareRuns()) {
    const datasets = [];
    const missing = [];
    for (const run of availableRuns) {
      if (!selectedRuns.has(run.run_id)) continue;
      const cache = manifestCache.get(run.run_id);
      const other = cache?.specById?.get(spec.id);
      if (!other?.series?.steps?.length) {
        missing.push(runLabel(run.run_id));
        continue;
      }
      const color = runColor(run.run_id);
      const isCurrent = run.run_id === runId;
      datasets.push({
        label: runLabel(run.run_id),
        data: pointsFromSeries(other.series),
        borderColor: color,
        backgroundColor: `${color}22`,
        borderWidth: isCurrent ? 2.5 : 1.75,
        borderDash: isCurrent ? [] : [6, 3],
        pointRadius: 0,
        tension: 0.2,
        fill: false,
      });
    }
    if (!datasets.length) {
      return {
        empty: true,
        emptyMessage: runsLoading
          ? "Loading setup data…"
          : missing.length
            ? `No matching series for: ${missing.join(", ")}`
            : "Select at least one setup above",
      };
    }
    return {
      legend: true,
      datasets,
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
          data: pointsFromSeries(m.series),
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
        label: spec.label,
        data: pointsFromSeries(spec.series),
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
  if (compareRuns && canCompareRuns()) return `${spec.label} · setup compare`;
  if (compareLayers && canCompareLayers(spec)) return `${spec.label} · layer compare`;
  return spec.label;
}

function chartInfoFor(spec, lineData = null) {
  const scaleLabel = curveScaleMode === "loglog" ? "log–log" : "linear";
  let base;
  if (compareRuns && canCompareRuns()) {
    const labels = availableRuns
      .filter((r) => selectedRuns.has(r.run_id))
      .map((r) => runLabel(r.run_id));
    base = `${spec.role} · ${labels.join(" vs ") || "no setup"} · every ${spec.every} steps`;
    if (lineData?.missingNote) base = `${base} · ${lineData.missingNote}`;
  } else if (compareLayers && canCompareLayers(spec)) {
    const labels = familyMembers(spec)
      .filter((m) => selectedLayers.has(m.layer))
      .sort((a, b) => a.layer - b.layer)
      .map((m) => `L${m.layer}`).join(", ") || "no layers selected";
    base = `${spec.role} · ${labels} · every ${spec.every} steps`;
  } else {
    base = `${spec.selector} · every ${spec.every} steps`;
  }
  return `${base} · ${scaleLabel}`;
}

function renderSpecDefinition(spec) {
  const el = document.getElementById("chartDef");
  if (!el) return;
  if (!spec) {
    el.hidden = true;
    el.innerHTML = "";
    return;
  }

  const direct = typeof buildSpecDirectFormula === "function"
    ? buildSpecDirectFormula(spec)
    : null;
  if (!direct?.tex) {
    el.hidden = true;
    el.innerHTML = "";
    return;
  }

  let mathHtml;
  if (window.katex) {
    try {
      mathHtml = katex.renderToString(direct.tex, {
        throwOnError: false,
        displayMode: false,
        output: "html",
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
  el.innerHTML = [
    `<div class="chart-def-row"><span class="chart-def-label">Definition</span>` +
      `<a class="chart-def-link" href="${href}" target="_blank" rel="noopener">All formulas</a></div>`,
    desc ? `<p class="chart-def-desc">${escapeHtml(desc)}</p>` : "",
    `<div class="chart-def-eq" title="${escapeHtml(direct.title)}">${mathHtml}</div>`,
  ].filter(Boolean).join("");
  el.hidden = false;
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

  const headers = lines[0].split(",").map((h) => h.trim());
  const col = (name) => headers.indexOf(name);
  const iterIdx = col("iter");
  const trainIdx = col("train_loss");
  const valIdx = col("val_loss");
  if (iterIdx < 0 || trainIdx < 0 || valIdx < 0) return null;

  const train = [];
  const val = [];
  for (let i = 1; i < lines.length; i += 1) {
    const cols = lines[i].split(",");
    const x = Number(cols[iterIdx]);
    const yt = Number(cols[trainIdx]);
    const yv = Number(cols[valIdx]);
    if (Number.isFinite(x) && Number.isFinite(yt)) train.push({ x, y: yt });
    if (Number.isFinite(x) && Number.isFinite(yv)) val.push({ x, y: yv });
  }
  if (!train.length && !val.length) return null;
  return { train, val };
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
  if (!xs.length) return { min: 0, max: 0 };
  return { min: Math.min(...xs), max: Math.max(...xs) };
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
  const bounds = lossDataBounds();
  let min = minEl.value === "" ? null : Number(minEl.value);
  let max = maxEl.value === "" ? null : Number(maxEl.value);
  if (min != null && !Number.isFinite(min)) min = null;
  if (max != null && !Number.isFinite(max)) max = null;
  if (min != null && max != null && min > max) [min, max] = [max, min];
  if (min != null && min < bounds.min) min = bounds.min;
  if (max != null && max > bounds.max) max = bounds.max;
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

function filterLossPoints(points) {
  const log = lossScaleMode === "loglog";
  return points
    .filter((p) => {
      if (lossStepMin != null && p.x < lossStepMin) return false;
      if (lossStepMax != null && p.x > lossStepMax) return false;
      if (log && !(p.y > 0)) return false;
      return Number.isFinite(p.x) && Number.isFinite(p.y);
    })
    .map((p) =>
      log
        ? { x: p.x + 1, y: p.y, _step: p.x }
        : { x: p.x, y: p.y, _step: p.x }
    );
}

function lossChartScaleOptions() {
  const axis = "rgba(255, 255, 255, 0.78)";
  const grid = "rgba(255, 255, 255, 0.12)";
  const log = lossScaleMode === "loglog";
  return {
    x: {
      type: log ? "logarithmic" : "linear",
      title: { display: true, text: log ? "Step (log)" : "Step", color: axis },
      ticks: {
        color: axis,
        ...(log
          ? {
              callback(value) {
                const step = Number(value) - 1;
                if (!Number.isFinite(step)) return "";
                return Math.abs(step - Math.round(step)) < 1e-9
                  ? String(Math.round(step))
                  : String(step);
              },
            }
          : {}),
      },
      grid: { color: grid },
    },
    y: {
      type: log ? "logarithmic" : "linear",
      title: { display: true, text: log ? "Loss (log)" : "Loss", color: axis },
      ticks: { color: axis },
      grid: { color: grid },
    },
  };
}

function lossChartOptions({ enablePan = true, onClick = null } = {}) {
  const opts = chartCommonOptions({
    legend: true,
    enablePan,
    onClick,
    scaleMode: lossScaleMode,
    yTitle: "Loss",
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
  const bounds = lossDataBounds();
  const lo = lossStepMin ?? bounds.min;
  const hi = lossStepMax ?? bounds.max;
  const modeLabel = lossViewMode === "both" ? "train + val" : lossViewMode;
  const scaleLabel = lossScaleMode === "loglog" ? "log–log" : "linear";
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
        ? `setup compare · no loss data (${missing.join(", ")})`
        : "setup compare · no loss data";
    }
    const names = logs.map((x) => x.label).join(" vs ");
    return (
      `${modeLabel} · ${scaleLabel} · setup compare · ${names} · step ${lo}–${hi}` +
      (missing.length ? ` · missing: ${missing.join(", ")}` : "")
    );
  }
  if (!logs.length) return "";
  const item = logs[0];
  const train = filterLossPoints(item.log.train);
  const val = filterLossPoints(item.log.val);
  const lastTrain = train.at(-1);
  const lastVal = val.at(-1);
  return (
    `${modeLabel} · ${scaleLabel} · step ${lo}–${hi} · ${train.length} train / ${val.length} val pts` +
    (lastTrain ? ` · train=${lastTrain.y.toFixed(4)}` : "") +
    (lastVal ? ` · val=${lastVal.y.toFixed(4)}` : "")
  );
}

function buildLossDatasets() {
  const logs = lossLogsForChart();
  if (!logs.length) return null;
  const datasets = [];

  if (compareLossRuns && canCompareRuns()) {
    for (const item of logs) {
      const color = runColor(item.run_id);
      const isCurrent = item.run_id === runId;
      if (lossViewMode === "both" || lossViewMode === "train") {
        datasets.push({
          label: `${item.label} · train`,
          data: filterLossPoints(item.log.train),
          borderColor: color,
          backgroundColor: `${color}22`,
          borderWidth: isCurrent ? 2.5 : 1.75,
          borderDash: [],
          pointRadius: 0,
          tension: 0.2,
          fill: false,
        });
      }
      if (lossViewMode === "both" || lossViewMode === "val") {
        datasets.push({
          label: `${item.label} · val`,
          data: filterLossPoints(item.log.val),
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
    if (!datasets.length) return null;
    if (!datasets.some((d) => (d.data || []).length)) return null;
    return datasets;
  }

  const item = logs[0];
  if (lossViewMode === "both" || lossViewMode === "train") {
    datasets.push({
      label: "Train loss",
      data: filterLossPoints(item.log.train),
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
      data: filterLossPoints(item.log.val),
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

function createLossChart(canvas, { enablePan = true, onClick = null } = {}) {
  const datasets = buildLossDatasets();
  if (!datasets) return null;
  return new Chart(canvas, {
    type: "line",
    data: { datasets },
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
}

function setLossOverlayChrome(show) {
  const el = document.getElementById("lossViewToggleFull");
  if (el) el.hidden = !show;
  const scale = document.getElementById("lossScaleToggleFull");
  if (scale) scale.hidden = !show;
  const range = document.getElementById("lossRangeFull");
  if (range) range.hidden = !show;
}

function setLossViewMode(mode) {
  if (!["both", "train", "val"].includes(mode)) return;
  lossViewMode = mode;
  updateLossViewButtons();
  renderLossChart();
  if (fullOverlayOpen && fullOverlayMode === "loss") renderFullLossChart();
}

function setLossScaleMode(mode) {
  if (!["linear", "loglog"].includes(mode)) return;
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
  lossChart = createLossChart(canvas);
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
  destroyFullChart();
  fullChart = createLossChart(canvas, {
    enablePan: false,
    onClick: fullscreenNoteClickHandler,
  });
  if (fullChart) bindFullscreenCanvasInteractions(canvas);
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
    // Do not destroy an existing healthy chart when a transient empty state appears.
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
    }),
  });
  bindFullscreenCanvasInteractions(canvas);
}

function openLossFullscreen() {
  if (!lossLog || lossLog.error || !buildLossDatasets()) return;
  fullOverlayMode = "loss";
  fullOverlayOpen = true;
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
  fullOverlayMode = "spec";
  fullOverlayOpen = true;
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
  fullOverlayOpen = false;
  fullOverlayMode = null;
  setLossOverlayChrome(false);
  setCurveOverlayChrome(false);
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

boot();
