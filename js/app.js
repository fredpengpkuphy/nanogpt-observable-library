const SOURCE_LABELS = {
  weight: "Weight",
  grad: "Gradient",
  update: "Update",
  activation: "Activation",
};

const SOURCE_COLORS = {
  weight: "#2563eb",
  grad: "#16a34a",
  update: "#d97706",
  activation: "#e11d48",
};

const ARCH_NODES = {
  embeddings: [
    { id: "wte", title: "Token Embed", subtitle: "wte", path: "transformer.wte" },
    { id: "wpe", title: "Pos Embed", subtitle: "wpe", path: "transformer.wpe" },
  ],
  block: [
    { id: "ln_1", suffix: "ln_1", title: "LayerNorm", subtitle: "ln_1", path: "ln_1" },
    { id: "c_attn", suffix: "attn.c_attn", title: "QKV Proj", subtitle: "attn.c_attn", path: "attn.c_attn" },
    { id: "c_proj_attn", suffix: "attn.c_proj", title: "Attn Out", subtitle: "attn.c_proj", path: "attn.c_proj" },
    { id: "ln_2", suffix: "ln_2", title: "LayerNorm", subtitle: "ln_2", path: "ln_2" },
    { id: "c_fc", suffix: "mlp.c_fc", title: "MLP Up", subtitle: "mlp.c_fc", path: "mlp.c_fc" },
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
let specById = new Map();
let specsByModule = new Map();
let activeLayer = 0;
let activeModuleId = null;
let activeSpecId = null;
let compareLayers = false;
let chart = null;

async function boot() {
  try {
    const latest = await fetchJson("data/latest.json");
    manifest = await fetchJson(`data/${latest.run_id}/manifest.json`);
    specById = new Map(manifest.specs.map((s) => [s.id, s]));
    specsByModule = groupSpecsByModule(manifest.specs);
    document.getElementById("compareLayers").addEventListener("change", (e) => {
      compareLayers = e.target.checked;
      const spec = activeSpecId ? specById.get(activeSpecId) : null;
      if (spec) renderChart(spec);
    });
    renderHeader();
    renderLayerTabs();
    renderArchitecture();
  } catch (err) {
    document.getElementById("architecture").innerHTML =
      `<div class="error">Failed to load viewer data: ${escapeHtml(err.message)}<br><br>` +
      `Run <code>python3 viewer/scripts/build_viewer_data.py</code> first.</div>`;
  }
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} (${res.status})`);
  return res.json();
}

function groupSpecsByModule(specs) {
  const map = new Map();
  for (const spec of specs) {
    if (!map.has(spec.ui_module)) map.set(spec.ui_module, []);
    map.get(spec.ui_module).push(spec);
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
  attnCol.appendChild(createModuleNode(moduleIdForBlock("attn.c_proj"), "Attn Out", "c_proj", moduleIdForBlock("attn.c_proj")));

  const mlpCol = section("MLP");
  mlpCol.appendChild(createModuleNode(moduleIdForBlock("ln_2"), "LayerNorm", "ln_2", moduleIdForBlock("ln_2")));
  mlpCol.appendChild(createModuleNode(moduleIdForBlock("mlp.c_fc"), "MLP Up", "c_fc", moduleIdForBlock("mlp.c_fc")));
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

  document.getElementById("moduleTitle").textContent = manifest.modules[moduleKey]?.label || title;
  const exampleSelector = specs[0]?.selector || `transformer.${moduleKey}`;
  document.getElementById("modulePath").textContent = exampleSelector;

  renderSpecGroups(specs);
  renderChart(null);
}

function renderSpecGroups(specs) {
  const groups = {};
  for (const spec of specs) {
    groups[spec.source_kind] = groups[spec.source_kind] || [];
    groups[spec.source_kind].push(spec);
  }

  const container = document.getElementById("specGroups");
  container.innerHTML = "";

  for (const kind of ["weight", "grad", "update", "activation"]) {
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
  if (!spec?.family_id || !manifest.families) return [];
  const family = manifest.families[spec.family_id];
  if (!family) return [];
  return family.spec_ids
    .map((id) => specById.get(id))
    .filter((s) => s && s.series?.steps?.length);
}

function canCompareLayers(spec) {
  return familyMembers(spec).length >= 2;
}

function updateCompareToggle(spec) {
  const wrap = document.getElementById("compareToggleWrap");
  const input = document.getElementById("compareLayers");
  const ok = canCompareLayers(spec);
  wrap.hidden = !ok;
  if (!ok) {
    compareLayers = false;
    input.checked = false;
  } else {
    input.checked = compareLayers;
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

function pointsFromSeries(series) {
  return series.steps.map((step, i) => ({ x: step, y: series.values[i] }));
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
    destroyChart();
    return;
  }

  destroyChart();

  // Overlay all layers that share the same nature (source/role/reduction/transforms).
  if (compareLayers && canCompareLayers(spec)) {
    const members = familyMembers(spec);
    titleEl.textContent = `${spec.label} · 各层叠加`;
    infoEl.textContent =
      `${spec.role} · ${members.length} layers · every ${spec.every} steps`;
    chart = new Chart(canvas, {
      type: "line",
      data: {
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
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "nearest", intersect: false },
        plugins: {
          legend: {
            display: true,
            position: "bottom",
            labels: { boxWidth: 12, color: "#475569", usePointStyle: true },
          },
          tooltip: {
            callbacks: {
              title: (items) => (items.length ? `step ${items[0].parsed.x}` : ""),
            },
          },
        },
        scales: chartScaleOptions(),
      },
    });
    return;
  }

  titleEl.textContent = spec.label;
  infoEl.textContent = `${spec.selector} · every ${spec.every} steps`;

  if (spec.series?.steps?.length) {
    const color = SOURCE_COLORS[spec.source_kind] || "#2563eb";
    chart = new Chart(canvas, {
      type: "line",
      data: {
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
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
        },
        scales: chartScaleOptions(),
      },
    });
    return;
  }

  if (spec.curve_png) {
    canvas.hidden = true;
    image.onload = () => image.classList.add("visible");
    image.onerror = () => {
      infoEl.textContent = "PNG curve not found yet — training may still be running.";
    };
    image.src = `data/${manifest.run_id}/${spec.curve_png}`;
    return;
  }

  infoEl.textContent = "No curve data yet.";
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
  const input = document.getElementById("compareLayers");
  if (input) input.checked = false;
  document.getElementById("compareToggleWrap").hidden = true;
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
