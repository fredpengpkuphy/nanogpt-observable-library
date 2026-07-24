const PAGE_SIZE = 40;

let catalog = [];
let filtered = [];
let page = 0;

async function bootReference() {
  const loading = document.getElementById("refLoading");
  try {
    const res = await fetch(`data/reference_catalog.json?t=${Date.now()}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`reference_catalog.json (${res.status})`);
    const data = await res.json();
    catalog = data.observables || [];
    document.getElementById("refSubtitle").textContent = data.run_id
      ? `Direct formulas from ${data.run_id}`
      : "Direct formula for each observable";
    fillFilters(catalog);
    wireControls();
    applyFilters();
    if (location.hash.startsWith("#obs-")) {
      jumpToId(decodeURIComponent(location.hash.slice(5)));
    }
  } catch (err) {
    if (loading) {
      loading.textContent =
        `Failed to load catalog: ${err.message}. ` +
        `Run: python scripts/build_reference_catalog.py`;
    }
  }
}

function fillFilters(items) {
  const sources = [...new Set(items.map((o) => o.source_kind).filter(Boolean))].sort();
  const reductions = [...new Set(items.map((o) => o.reduction).filter(Boolean))].sort();
  const srcSel = document.getElementById("refSourceFilter");
  const redSel = document.getElementById("refReductionFilter");
  for (const s of sources) {
    const opt = document.createElement("option");
    opt.value = s;
    opt.textContent = s;
    srcSel.appendChild(opt);
  }
  for (const r of reductions) {
    const opt = document.createElement("option");
    opt.value = r;
    opt.textContent = r;
    redSel.appendChild(opt);
  }
}

function wireControls() {
  const search = document.getElementById("refSearch");
  const src = document.getElementById("refSourceFilter");
  const red = document.getElementById("refReductionFilter");
  let timer = null;
  search.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      page = 0;
      applyFilters();
    }, 120);
  });
  src.addEventListener("change", () => {
    page = 0;
    applyFilters();
  });
  red.addEventListener("change", () => {
    page = 0;
    applyFilters();
  });
  document.getElementById("refPrevPage").addEventListener("click", () => {
    if (page > 0) {
      page -= 1;
      renderPage();
    }
  });
  document.getElementById("refNextPage").addEventListener("click", () => {
    const maxPage = Math.max(0, Math.ceil(filtered.length / PAGE_SIZE) - 1);
    if (page < maxPage) {
      page += 1;
      renderPage();
    }
  });
  const pageInput = document.getElementById("refPageInput");
  const jumpToTypedPage = () => {
    const maxPage = Math.max(0, Math.ceil(filtered.length / PAGE_SIZE) - 1);
    let n = Number(pageInput.value);
    if (!Number.isFinite(n)) {
      pageInput.value = String(page + 1);
      return;
    }
    n = Math.round(n);
    n = Math.min(Math.max(n, 1), maxPage + 1);
    page = n - 1;
    renderPage();
  };
  pageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      jumpToTypedPage();
      pageInput.blur();
    }
  });
  pageInput.addEventListener("change", jumpToTypedPage);
  pageInput.addEventListener("focus", () => pageInput.select());
}

function applyFilters() {
  const q = (document.getElementById("refSearch").value || "").trim().toLowerCase();
  const src = document.getElementById("refSourceFilter").value;
  const red = document.getElementById("refReductionFilter").value;
  filtered = catalog.filter((o) => {
    if (src && o.source_kind !== src) return false;
    if (red && o.reduction !== red) return false;
    if (!q) return true;
    const hay = [
      o.id,
      o.label,
      o.selector,
      o.source_kind,
      o.reduction,
      o.ui_module,
      o.role,
      ...(o.transforms || []),
      ...(o.temporal || []).flat(Infinity).map(String),
    ]
      .join(" ")
      .toLowerCase();
    return hay.includes(q);
  });
  renderPage();
}

function jumpToId(id) {
  const idx = catalog.findIndex((o) => o.id === id);
  if (idx < 0) return;
  const item = catalog[idx];
  document.getElementById("refSearch").value = item.selector || item.id;
  document.getElementById("refSourceFilter").value = "";
  document.getElementById("refReductionFilter").value = "";
  applyFilters();
  const fidx = filtered.findIndex((o) => o.id === id);
  if (fidx >= 0) {
    page = Math.floor(fidx / PAGE_SIZE);
    renderPage();
    requestAnimationFrame(() => {
      const el = document.getElementById(`obs-${id}`);
      el?.scrollIntoView({ block: "center" });
      el?.classList.add("ref-obs-highlight");
    });
  }
}

function renderPage() {
  const root = document.getElementById("refCatalog");
  const total = filtered.length;
  const maxPage = Math.max(0, Math.ceil(total / PAGE_SIZE) - 1);
  if (page > maxPage) page = maxPage;
  const start = page * PAGE_SIZE;
  const slice = filtered.slice(start, start + PAGE_SIZE);

  const end = Math.min(start + PAGE_SIZE, total);
  const shownStart = total === 0 ? 0 : start + 1;
  document.getElementById("refCount").textContent =
    total === 0
      ? "No matches"
      : total === catalog.length
        ? `${shownStart}–${end} of ${total}`
        : `${shownStart}–${end} of ${total} · ${catalog.length} total`;
  const pageInput = document.getElementById("refPageInput");
  const pageTotal = document.getElementById("refPageTotal");
  if (pageInput && document.activeElement !== pageInput) {
    pageInput.value = total === 0 ? "" : String(page + 1);
    pageInput.max = String(maxPage + 1);
    pageInput.disabled = total === 0;
  }
  if (pageTotal) pageTotal.textContent = total === 0 ? "—" : String(maxPage + 1);
  document.getElementById("refPrevPage").disabled = page <= 0;
  document.getElementById("refNextPage").disabled = page >= maxPage;

  if (!slice.length) {
    root.innerHTML = `<p class="ref-loading">No matching observables.</p>`;
    return;
  }

  root.innerHTML = "";
  for (const obs of slice) {
    root.appendChild(renderCard(obs));
  }
}

function renderCard(obs) {
  const article = document.createElement("article");
  article.className = "ref-obs";
  article.id = `obs-${obs.id}`;

  const direct =
    typeof buildSpecDirectFormula === "function" ? buildSpecDirectFormula(obs) : null;
  const tex = direct?.tex || `y_t=\\mathrm{${obs.reduction}}(\\cdots)`;

  let mathHtml;
  if (window.katex) {
    try {
      mathHtml = katex.renderToString(tex, {
        throwOnError: true,
        displayMode: true,
        fleqn: true,
        output: "htmlAndMathml",
      });
    } catch (_) {
      mathHtml = `<code>${escapeHtml(tex)}</code>`;
    }
  } else {
    mathHtml = `<code>${escapeHtml(tex)}</code>`;
  }

  const transforms = (obs.transforms || []).length
    ? (obs.transforms || []).join(" → ")
    : "—";
  const temporal = (obs.temporal || []).length
    ? (obs.temporal || [])
        .map((entry) =>
          typeof formatTemporalEntry === "function"
            ? formatTemporalEntry(entry)
            : String(entry)
        )
        .join(" → ")
    : (
        typeof temporalOpsFromCanonicalId === "function"
          ? temporalOpsFromCanonicalId(obs).join(" → ")
          : ""
      ) || "—";
  const layer =
    obs.layer === null || obs.layer === undefined ? "—" : `L${obs.layer}`;

  const description =
    (typeof buildSpecPlainDescription === "function" && buildSpecPlainDescription(obs)) ||
    direct?.description ||
    "";
  const notation = Array.isArray(direct?.notation)
    ? direct.notation
    : (typeof buildSpecNotation === "function" ? buildSpecNotation(obs) : []);
  const notationHtml = notation.length
    ? `<div class="formula-notation ref-formula-notation">` +
      `<div class="formula-notation-title">Notation</div>` +
      `<dl>${notation.map((item) =>
        `<div><dt aria-label="${escapeHtml(item.symbol)}">${renderNotationSymbolHtml(item)}</dt>` +
        `<dd>${escapeHtml(item.meaning)}</dd></div>`
      ).join("")}</dl></div>`
    : "";

  article.innerHTML = `
    <div class="ref-obs-head">
      <h3 class="ref-obs-title">${escapeHtml(obs.selector || obs.id)}</h3>
      <div class="ref-obs-tags">
        <span class="ref-tag">${escapeHtml(obs.source_kind)}</span>
        <span class="ref-tag">${escapeHtml(obs.reduction)}</span>
        <span class="ref-tag">${escapeHtml(layer)}</span>
      </div>
    </div>
    <p class="ref-obs-desc">${escapeHtml(description)}</p>
    <div class="ref-obs-formula">${mathHtml}</div>
    ${notationHtml}
    <div class="ref-obs-meta">
      <span>${escapeHtml(obs.label || "")}</span>
      <span>transforms: ${escapeHtml(transforms)}</span>
      <span>temporal: ${escapeHtml(temporal)}</span>
    </div>
  `;
  return article;
}

function renderNotationSymbolHtml(item) {
  const fallback = String(item?.symbol || "");
  const tex =
    item?.tex ||
    (typeof notationSymbolTex === "function" ? notationSymbolTex(fallback) : fallback);
  if (window.katex && tex) {
    try {
      return katex.renderToString(tex, {
        throwOnError: true,
        displayMode: false,
        output: "htmlAndMathml",
      });
    } catch (_) {
      /* use the readable source symbol below */
    }
  }
  return escapeHtml(fallback);
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

document.addEventListener("DOMContentLoaded", async () => {
  if (typeof MaintenanceGate !== "undefined") {
    const ok = await MaintenanceGate.wire();
    if (!ok) return;
  }
  await bootReference();
});
