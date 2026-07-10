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
  weight: "#2563eb",
  grad: "#16a34a",
  update: "#d97706",
  activation: "#e11d48",
  preactivation: "#be185d",
  gelu_activation: "#c026d3",
  attention: "#0f766e",
  logits: "#4338ca",
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
  block: [
    { id: "ln_1", suffix: "ln_1", title: "LayerNorm", subtitle: "ln_1", path: "ln_1" },
    { id: "c_attn", suffix: "attn.c_attn", title: "QKV Proj", subtitle: "attn.c_attn", path: "attn.c_attn" },
    { id: "attn", suffix: "attn", title: "Attention", subtitle: "attn · entropy", path: "attn" },
    { id: "c_proj_attn", suffix: "attn.c_proj", title: "Attn Out", subtitle: "attn.c_proj", path: "attn.c_proj" },
    { id: "ln_2", suffix: "ln_2", title: "LayerNorm", subtitle: "ln_2", path: "ln_2" },
    { id: "c_fc", suffix: "mlp.c_fc", title: "MLP Up", subtitle: "mlp.c_fc", path: "mlp.c_fc" },
    { id: "gelu", suffix: "mlp.gelu", title: "GELU", subtitle: "mlp.gelu · massive", path: "mlp.gelu" },
    { id: "c_proj_mlp", suffix: "mlp.c_proj", title: "MLP Down", subtitle: "mlp.c_proj", path: "mlp.c_proj" },
  ],
  head: [
    { id: "ln_f", title: "Final LN", subtitle: "ln_f", path: "transformer.ln_f" },
    { id: "lm_head", title: "LM Head", subtitle: "lm_head", path: "lm_head", optional: true },
  ],
};

// Distinct hues for L0..L11 overlays (readable on white).
const LAYER_COLORS = [
  "#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2",
  "#db2777", "#65a30d", "#ea580c", "#4f46e5", "#0d9488", "#b45309",
];

let manifest = null;
let runId = null;
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

async function boot() {
  runId = new URLSearchParams(location.search).get("run");
  if (!runId) {
    window.location.href = "index.html";
    return;
  }

  try {
    if (window.Chart && window.ChartZoom) {
      try { Chart.register(window.ChartZoom); } catch (_) { /* already registered */ }
    }
    manifest = await fetchJson(`data/${runId}/manifest.json`);
    specById = new Map(manifest.specs.map((s) => [s.id, s]));
    specsByModule = groupSpecsByModule(manifest.specs);
    specsByFamily = groupSpecsByFamily(manifest.specs);
    document.getElementById("compareLayers").addEventListener("change", (e) => {
      compareLayers = e.target.checked;
      const spec = activeSpecId ? specById.get(activeSpecId) : null;
      if (compareLayers && spec) ensureDefaultSelectedLayers(spec);
      if (spec) renderChart(spec);
    });
    document.getElementById("resetZoomBtn").addEventListener("click", resetChartZoom);
    document.getElementById("lossResetZoomBtn").addEventListener("click", () => lossChart?.resetZoom?.());
    document.getElementById("lossExpandBtn").addEventListener("click", openLossFullscreen);
    document.querySelectorAll(".loss-view-btn").forEach((btn) => {
      btn.addEventListener("click", () => setLossViewMode(btn.dataset.mode));
    });
    document.getElementById("expandBtn").addEventListener("click", openFullscreen);
    document.getElementById("fullCloseBtn").addEventListener("click", closeFullscreen);
    document.getElementById("fullResetZoomBtn").addEventListener("click", () => fullChart?.resetZoom?.());
    document.getElementById("chartOverlay").addEventListener("click", (e) => {
      if (e.target.id === "chartOverlay") closeFullscreen();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && fullOverlayOpen) closeFullscreen();
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
    renderHeader();
    lossLog = await loadLossLog();
    renderLossChart();
    renderLayerTabs();
    renderArchitecture();
  } catch (err) {
    document.getElementById("architecture").innerHTML =
      `<div class="error">Failed to load viewer data: ${escapeHtml(err.message)}<br><br>` +
      `Run <code>python3 viewer/scripts/build_viewer_data.py --clean</code> or ` +
      `<a href="index.html">choose another dataset</a>.</div>`;
  }
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

function destroyLossChart() {
  if (lossChart) {
    lossChart.destroy();
    lossChart = null;
  }
}

function lossInfoText() {
  if (!lossLog || lossLog.error) return "";
  const lastTrain = lossLog.train.at(-1);
  const lastVal = lossLog.val.at(-1);
  const maxStep = Math.max(lastTrain?.x ?? 0, lastVal?.x ?? 0);
  const modeLabel = lossViewMode === "both" ? "train + val" : lossViewMode;
  return (
    `${modeLabel} · ${lossLog.train.length} train · ${lossLog.val.length} val · step 0–${maxStep}` +
    (lastTrain ? ` · train=${lastTrain.y.toFixed(4)}` : "") +
    (lastVal ? ` · val=${lastVal.y.toFixed(4)}` : "")
  );
}

function buildLossDatasets() {
  if (!lossLog || lossLog.error) return null;
  const datasets = [];
  if (lossViewMode === "both" || lossViewMode === "train") {
    datasets.push({
      label: "Train loss",
      data: lossLog.train,
      borderColor: "#2563eb",
      backgroundColor: "rgba(37, 99, 235, 0.12)",
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.2,
      fill: false,
    });
  }
  if (lossViewMode === "both" || lossViewMode === "val") {
    datasets.push({
      label: "Val loss",
      data: lossLog.val,
      borderColor: "#dc2626",
      backgroundColor: "rgba(220, 38, 38, 0.12)",
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.2,
      fill: false,
    });
  }
  return datasets.length ? datasets : null;
}

function createLossChart(canvas) {
  const datasets = buildLossDatasets();
  if (!datasets) return null;
  return new Chart(canvas, {
    type: "line",
    data: { datasets },
    options: chartCommonOptions({ legend: true }),
  });
}

function updateLossViewButtons() {
  document.querySelectorAll(".loss-view-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === lossViewMode);
  });
}

function setLossOverlayChrome(show) {
  const el = document.getElementById("lossViewToggleFull");
  if (el) el.hidden = !show;
}

function setLossViewMode(mode) {
  if (!["both", "train", "val"].includes(mode)) return;
  lossViewMode = mode;
  updateLossViewButtons();
  renderLossChart();
  if (fullOverlayOpen && fullOverlayMode === "loss") renderFullLossChart();
}

function renderLossChart() {
  const section = document.getElementById("lossSection");
  const infoEl = document.getElementById("lossInfo");
  const canvas = document.getElementById("lossChart");
  destroyLossChart();
  updateLossViewButtons();

  if (lossLog?.error) {
    section.hidden = false;
    infoEl.textContent = "未找到 eval_loss_log.csv（请确认该文件已 push 到仓库）";
    return;
  }

  if (!lossLog) {
    section.hidden = true;
    return;
  }

  section.hidden = false;
  infoEl.textContent = lossInfoText();

  lossChart = createLossChart(canvas);
  if (lossChart) canvas.ondblclick = () => lossChart?.resetZoom?.();
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

function renderHeader() {
  const m = manifest.model;
  document.getElementById("runSubtitle").textContent =
    `${m.name} · ${m.n_layer} layers · ${m.n_embd} dim · ${manifest.n_specs} observables`;
  document.getElementById("runBadge").textContent = manifest.run_id;
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
    const uiId = node.id;
    headRow.appendChild(createModuleNode(uiId, node.title, node.subtitle, uiId, node.optional));
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
    if (optional) {
      el.querySelector?.(".subtitle");
    }
  }

  el.innerHTML = `
    ${count > 0 ? `<span class="count">${count}</span>` : ""}
    <div class="title">${escapeHtml(title)}</div>
    <div class="subtitle">${escapeHtml(subtitle)}${count === 0 ? " · no data" : ""}</div>
  `;
  return el;
}

function selectModule(moduleKey, title, subtitle) {
  activeModuleId = moduleKey;
  activeSpecId = null;
  renderArchitecture();

  const specs = specsByModule.get(moduleKey) || [];
  document.getElementById("detailEmpty").style.display = "none";
  const detail = document.getElementById("detailContent");
  detail.classList.add("visible");

  document.getElementById("moduleTitle").textContent = friendlyModuleLabel(moduleKey, title);
  const exampleSelector = specs[0]?.selector || `transformer.${moduleKey}`;
  document.getElementById("modulePath").textContent = exampleSelector;

  renderSpecGroups(specs);
  renderChart(null);
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
  const wrap = document.getElementById("compareToggleWrap");
  const input = document.getElementById("compareLayers");
  const pickWrap = document.getElementById("layerPickWrap");
  const ok = canCompareLayers(spec);
  wrap.hidden = !ok;
  if (!ok) {
    compareLayers = false;
    input.checked = false;
    pickWrap.hidden = true;
    return;
  }
  input.checked = compareLayers;
  if (compareLayers) {
    ensureDefaultSelectedLayers(spec);
    renderLayerPicker(spec);
    pickWrap.hidden = false;
  } else {
    pickWrap.hidden = true;
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

function chartScaleOptions() {
  return {
    x: {
      type: "linear",
      title: { display: true, text: "Step", color: "#64748b" },
      ticks: { color: "#64748b" },
      grid: { color: "rgba(30,64,128,0.08)" },
    },
    y: {
      title: { display: true, text: "Value", color: "#64748b" },
      ticks: { color: "#64748b" },
      grid: { color: "rgba(30,64,128,0.08)" },
    },
  };
}

function zoomPluginOptions() {
  return {
    zoom: {
      wheel: { enabled: true, speed: 0.1 },
      pinch: { enabled: true },
      mode: "xy",
      drag: { enabled: false },
    },
    pan: {
      enabled: true,
      mode: "xy",
      modifierKey: null,
    },
    limits: {
      x: { min: "original", max: "original" },
      y: { min: "original", max: "original" },
    },
  };
}

function chartCommonOptions({ legend = false } = {}) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "nearest", intersect: false },
    plugins: {
      legend: legend
        ? {
            display: true,
            position: "bottom",
            labels: { boxWidth: 12, color: "#475569", usePointStyle: true },
          }
        : { display: false },
      tooltip: {
        callbacks: {
          title: (items) => (items.length ? `step ${items[0].parsed.x}` : ""),
        },
      },
      zoom: zoomPluginOptions(),
    },
    scales: chartScaleOptions(),
  };
}

function attachDoubleClickReset(canvas) {
  canvas.ondblclick = () => resetChartZoom();
}

function resetChartZoom() {
  if (chart?.resetZoom) chart.resetZoom();
}

function setChartChrome(hasSeriesChart) {
  document.getElementById("resetZoomBtn").hidden = !hasSeriesChart;
  document.getElementById("chartHint").hidden = !hasSeriesChart;
  document.getElementById("expandBtn").hidden = !hasSeriesChart;
}

function pointsFromSeries(series) {
  return series.steps.map((step, i) => ({ x: step, y: series.values[i] }));
}

// Build the Chart.js datasets for the currently selected spec/state.
// Returns { datasets, legend } for a line chart, { empty:true } when compare
// mode has no layers selected, or null when there is no plottable series.
function buildLineDatasets(spec) {
  if (compareLayers && canCompareLayers(spec)) {
    const members = familyMembers(spec)
      .filter((m) => selectedLayers.has(m.layer))
      .sort((a, b) => a.layer - b.layer);
    if (!members.length) return { empty: true };
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
    const color = SOURCE_COLORS[spec.source_kind] || "#2563eb";
    return {
      legend: false,
      datasets: [{
        label: spec.label,
        data: pointsFromSeries(spec.series),
        borderColor: color,
        backgroundColor: `${color}22`,
        borderWidth: 2,
        pointRadius: 2,
        tension: 0.25,
        fill: true,
      }],
    };
  }
  return null;
}

function chartTitleFor(spec) {
  if (compareLayers && canCompareLayers(spec)) return `${spec.label} · 层对比`;
  return spec.label;
}

function renderChart(spec) {
  const titleEl = document.getElementById("chartTitle");
  const infoEl = document.getElementById("chartInfo");
  const canvas = document.getElementById("curveChart");
  const image = document.getElementById("curveImage");

  canvas.hidden = false;
  image.classList.remove("visible");
  image.removeAttribute("src");
  updateCompareToggle(spec);

  if (!spec) {
    titleEl.textContent = "Select an observable";
    infoEl.textContent = "";
    setChartChrome(false);
    destroyChart();
    return;
  }

  destroyChart();

  // Titles/subtitles.
  if (compareLayers && canCompareLayers(spec)) {
    const selected = familyMembers(spec)
      .filter((m) => selectedLayers.has(m.layer))
      .sort((a, b) => a.layer - b.layer);
    const labels = selected.map((m) => `L${m.layer}`).join(", ") || "未选层";
    titleEl.textContent = chartTitleFor(spec);
    infoEl.textContent = `${spec.role} · ${labels} · every ${spec.every} steps`;
  } else {
    titleEl.textContent = spec.label;
    infoEl.textContent = `${spec.selector} · every ${spec.every} steps`;
  }

  const lineData = buildLineDatasets(spec);

  if (lineData?.empty) {
    setChartChrome(false);
    infoEl.textContent = "请在上方勾选要对比的层";
    return;
  }

  if (lineData) {
    chart = new Chart(canvas, {
      type: "line",
      data: { datasets: lineData.datasets },
      options: chartCommonOptions({ legend: lineData.legend }),
    });
    attachDoubleClickReset(canvas);
    setChartChrome(true);
    if (fullOverlayOpen) renderFullChart(spec);
    return;
  }

  setChartChrome(false);

  if (spec.curve_png) {
    canvas.hidden = true;
    image.onload = () => image.classList.add("visible");
    image.onerror = () => {
      infoEl.textContent = "PNG curve not found yet — training may still be running.";
    };
    image.src = `data/${runId}/${spec.curve_png}`;
    return;
  }

  infoEl.textContent = "No curve data yet.";
}

let fullChart = null;
let fullOverlayOpen = false;
let fullOverlayMode = null;

function renderFullLossChart() {
  const canvas = document.getElementById("curveChartFull");
  const titleEl = document.getElementById("fullChartTitle");
  const infoEl = document.getElementById("fullChartInfo");
  if (fullChart) {
    fullChart.destroy();
    fullChart = null;
  }
  titleEl.textContent = "Training Loss";
  infoEl.textContent = lossInfoText();
  fullChart = createLossChart(canvas);
  if (fullChart) canvas.ondblclick = () => fullChart?.resetZoom?.();
}

function renderFullChart(spec) {
  const canvas = document.getElementById("curveChartFull");
  const titleEl = document.getElementById("fullChartTitle");
  const infoEl = document.getElementById("fullChartInfo");
  if (fullChart) {
    fullChart.destroy();
    fullChart = null;
  }
  const lineData = buildLineDatasets(spec);
  titleEl.textContent = chartTitleFor(spec);
  if (compareLayers && canCompareLayers(spec)) {
    const labels = familyMembers(spec)
      .filter((m) => selectedLayers.has(m.layer))
      .sort((a, b) => a.layer - b.layer)
      .map((m) => `L${m.layer}`).join(", ") || "未选层";
    infoEl.textContent = `${spec.role} · ${labels} · every ${spec.every} steps`;
  } else {
    infoEl.textContent = `${spec.selector} · every ${spec.every} steps`;
  }
  if (!lineData || lineData.empty) {
    infoEl.textContent = lineData?.empty ? "请在上方勾选要对比的层" : "No curve data yet.";
    return;
  }
  fullChart = new Chart(canvas, {
    type: "line",
    data: { datasets: lineData.datasets },
    options: chartCommonOptions({ legend: lineData.legend }),
  });
  canvas.ondblclick = () => fullChart?.resetZoom?.();
}

function openLossFullscreen() {
  if (!lossLog || lossLog.error || !buildLossDatasets()) return;
  fullOverlayMode = "loss";
  fullOverlayOpen = true;
  setLossOverlayChrome(true);
  const overlay = document.getElementById("chartOverlay");
  overlay.hidden = false;
  overlay.classList.add("visible");
  overlay.setAttribute("aria-hidden", "false");
  renderFullLossChart();
}

function openFullscreen() {
  const spec = activeSpecId ? specById.get(activeSpecId) : null;
  if (!spec) return;
  fullOverlayMode = "spec";
  fullOverlayOpen = true;
  setLossOverlayChrome(false);
  const overlay = document.getElementById("chartOverlay");
  overlay.hidden = false;
  overlay.classList.add("visible");
  overlay.setAttribute("aria-hidden", "false");
  renderFullChart(spec);
}

function closeFullscreen() {
  fullOverlayOpen = false;
  fullOverlayMode = null;
  setLossOverlayChrome(false);
  const overlay = document.getElementById("chartOverlay");
  overlay.classList.remove("visible");
  overlay.setAttribute("aria-hidden", "true");
  overlay.hidden = true;
  if (fullChart) {
    fullChart.destroy();
    fullChart = null;
  }
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
  selectedLayers = new Set();
  const input = document.getElementById("compareLayers");
  if (input) input.checked = false;
  document.getElementById("compareToggleWrap").hidden = true;
  document.getElementById("layerPickWrap").hidden = true;
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
