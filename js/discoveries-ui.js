/**
 * Discoveries page: long-form public findings with combined curve references,
 * nested replies, and curator deletion controls.
 */
const DiscoveriesUI = (() => {
  const MAX_REFERENCES = 6;
  const CURVE_COLORS = [
    "#9fd4e4",
    "#e8c888",
    "#ef8a8a",
    "#b8d982",
    "#c7a6e8",
    "#f0a4cf",
  ];
  const CURVE_DASHES = [[], [8, 4], [2, 3], [10, 4, 2, 4], [5, 3], [12, 4]];
  const manifestCache = new Map();
  const draftCharts = new Set();
  const feedCharts = new Set();
  let runs = [];
  let draftReferences = [];
  let discoveries = [];
  let isAdmin = false;

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function formatTime(value) {
    if (!value) return "";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? "" : date.toLocaleString();
  }

  function validStep(value) {
    return Number.isInteger(value) && value >= 0 && value <= 1_000_000_000_000;
  }

  function displaySpec(spec) {
    if (typeof observableDisplayLabel === "function") {
      return observableDisplayLabel(spec);
    }
    return spec?.label || spec?.id || "Observable";
  }

  async function fetchJson(url) {
    const separator = url.includes("?") ? "&" : "?";
    const response = await fetch(`${url}${separator}t=${Date.now()}`, {
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`Could not load ${url}`);
    return response.json();
  }

  function specHasSeries(spec) {
    const steps = Array.isArray(spec?.series?.steps) ? spec.series.steps : [];
    const values = Array.isArray(spec?.series?.values) ? spec.series.values : [];
    const count = Math.min(steps.length, values.length);
    for (let index = 0; index < count; index += 1) {
      if (validStep(steps[index]) && Number.isFinite(values[index])) return true;
    }
    return false;
  }

  function recordedSteps(spec) {
    const steps = Array.isArray(spec?.series?.steps) ? spec.series.steps : [];
    const values = Array.isArray(spec?.series?.values) ? spec.series.values : [];
    const result = [];
    const count = Math.min(steps.length, values.length);
    for (let index = 0; index < count; index += 1) {
      if (validStep(steps[index]) && Number.isFinite(values[index])) {
        result.push(steps[index]);
      }
    }
    return result;
  }

  async function loadManifest(runId) {
    if (manifestCache.has(runId)) return manifestCache.get(runId);
    const promise = fetchJson(`data/${encodeURIComponent(runId)}/manifest.json`).then(
      (manifest) => {
        manifest._specMap = new Map(
          (manifest.specs || []).filter(specHasSeries).map((spec) => [spec.id, spec])
        );
        return manifest;
      }
    );
    manifestCache.set(runId, promise);
    try {
      return await promise;
    } catch (err) {
      manifestCache.delete(runId);
      throw err;
    }
  }

  function setReferenceStatus(message, isError = false) {
    const status = document.getElementById("discoveryReferenceStatus");
    if (!status) return;
    status.textContent = message || "";
    status.classList.toggle("is-error", !!isError);
  }

  function setFormStatus(message, isError = false) {
    const status = document.getElementById("discoveryFormStatus");
    if (!status) return;
    status.textContent = message || "";
    status.classList.toggle("is-error", !!isError);
  }

  function option(value, label) {
    return `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`;
  }

  async function populateRuns() {
    const index = await fetchJson("data/index.json");
    runs = Array.isArray(index.runs) ? index.runs : [];
    const select = document.getElementById("discoveryRun");
    select.innerHTML = runs
      .map((run) => option(run.run_id, run.label || run.run_id))
      .join("");
    if (!runs.length) {
      select.innerHTML = option("", "No setups available");
      return;
    }
    await populateModules();
  }

  async function populateModules() {
    const runSelect = document.getElementById("discoveryRun");
    const moduleSelect = document.getElementById("discoveryModule");
    const specSelect = document.getElementById("discoverySpec");
    const addButton = document.getElementById("discoveryAddReference");
    const selectedRun = runSelect.value;
    moduleSelect.disabled = true;
    specSelect.disabled = true;
    addButton.disabled = true;
    moduleSelect.innerHTML = option("", "Loading model positions…");
    specSelect.innerHTML = option("", "Choose a model position first");
    setReferenceStatus("Loading recorded observables…");
    try {
      const manifest = await loadManifest(selectedRun);
      const modules = Object.entries(manifest.modules || {})
        .map(([id, module]) => ({
          id,
          label: module.label || id,
          specIds: Array.isArray(module.spec_ids) ? module.spec_ids : [],
        }))
        .filter((module) =>
          module.specIds.some((specId) => manifest._specMap.has(specId))
        )
        .sort((a, b) =>
          a.label.localeCompare(b.label, undefined, {
            numeric: true,
            sensitivity: "base",
          })
        );
      moduleSelect.innerHTML = modules
        .map((module) => option(module.id, module.label))
        .join("");
      moduleSelect.disabled = !modules.length;
      setReferenceStatus("");
      await populateSpecs();
    } catch (err) {
      moduleSelect.innerHTML = option("", "Could not load this setup");
      setReferenceStatus(err.message || String(err), true);
    }
  }

  async function populateSpecs() {
    const runId = document.getElementById("discoveryRun").value;
    const moduleId = document.getElementById("discoveryModule").value;
    const specSelect = document.getElementById("discoverySpec");
    const addButton = document.getElementById("discoveryAddReference");
    specSelect.disabled = true;
    addButton.disabled = true;
    try {
      const manifest = await loadManifest(runId);
      const module = manifest.modules?.[moduleId];
      const specs = (module?.spec_ids || [])
        .map((specId) => manifest._specMap.get(specId))
        .filter(Boolean)
        .sort((a, b) =>
          displaySpec(a).localeCompare(displaySpec(b), undefined, {
            numeric: true,
            sensitivity: "base",
          })
        );
      specSelect.innerHTML = specs
        .map((spec) => option(spec.id, displaySpec(spec)))
        .join("");
      specSelect.disabled = !specs.length;
      addButton.disabled = !specs.length;
      updateStepBounds();
    } catch (err) {
      specSelect.innerHTML = option("", "Could not load observables");
      setReferenceStatus(err.message || String(err), true);
    }
  }

  async function updateStepBounds() {
    const runId = document.getElementById("discoveryRun").value;
    const specId = document.getElementById("discoverySpec").value;
    if (!runId || !specId) return;
    const manifest = await loadManifest(runId);
    const spec = manifest._specMap.get(specId);
    const steps = recordedSteps(spec);
    if (!steps.length) return;
    const min = Math.min(...steps);
    const max = Math.max(...steps);
    const start = document.getElementById("discoveryStepStart");
    const end = document.getElementById("discoveryStepEnd");
    start.min = String(min);
    start.max = String(max);
    end.min = String(min);
    end.max = String(max);
    start.value = String(min);
    end.value = String(max);
  }

  function nearestRecordedRange(steps, requestedStart, requestedEnd) {
    const ordered = [...new Set(steps.filter(validStep))].sort((a, b) => a - b);
    const start = ordered.find((step) => step >= requestedStart);
    let end = null;
    for (let index = ordered.length - 1; index >= 0; index -= 1) {
      if (ordered[index] <= requestedEnd) {
        end = ordered[index];
        break;
      }
    }
    if (!validStep(start) || !validStep(end) || end <= start) return null;
    return { stepStart: start, stepEnd: end };
  }

  async function addDraftReference() {
    if (draftReferences.length >= MAX_REFERENCES) {
      setReferenceStatus(`A chart can contain up to ${MAX_REFERENCES} curves.`, true);
      return;
    }
    const runSelect = document.getElementById("discoveryRun");
    const moduleSelect = document.getElementById("discoveryModule");
    const specSelect = document.getElementById("discoverySpec");
    const requestedStart = Number(document.getElementById("discoveryStepStart").value);
    const requestedEnd = Number(document.getElementById("discoveryStepEnd").value);
    if (!validStep(requestedStart) || !validStep(requestedEnd) || requestedEnd <= requestedStart) {
      setReferenceStatus("Choose a valid range whose end step is after its start.", true);
      return;
    }
    const manifest = await loadManifest(runSelect.value);
    const spec = manifest._specMap.get(specSelect.value);
    const range = nearestRecordedRange(
      recordedSteps(spec),
      requestedStart,
      requestedEnd
    );
    if (!range) {
      setReferenceStatus("That range does not contain at least two recorded steps.", true);
      return;
    }
    const run = runs.find((item) => item.run_id === runSelect.value);
    const reference = {
      runId: runSelect.value,
      runLabel: run?.label || runSelect.value,
      moduleId: moduleSelect.value,
      moduleLabel: moduleSelect.selectedOptions[0]?.textContent || moduleSelect.value,
      specId: specSelect.value,
      specLabel: displaySpec(spec),
      ...range,
    };
    const duplicate = draftReferences.some(
      (item) =>
        item.runId === reference.runId &&
        item.specId === reference.specId &&
        item.stepStart === reference.stepStart &&
        item.stepEnd === reference.stepEnd
    );
    if (duplicate) {
      setReferenceStatus("That exact curve region is already attached.", true);
      return;
    }
    draftReferences.push(reference);
    document.getElementById("discoveryStepStart").value = String(range.stepStart);
    document.getElementById("discoveryStepEnd").value = String(range.stepEnd);
    setReferenceStatus("Curve added to the combined chart.");
    renderDraftReferences();
  }

  function destroyCharts(set) {
    for (const chart of set) {
      try {
        chart.destroy();
      } catch (_) {}
    }
    set.clear();
  }

  function referenceHeading(reference) {
    return [
      reference.runLabel || reference.runId,
      reference.moduleLabel || reference.moduleId,
      reference.specLabel || reference.specId,
    ]
      .filter(Boolean)
      .join(" · ");
  }

  function referenceBounds(references) {
    const starts = references.map((reference) => reference.stepStart).filter(validStep);
    const ends = references.map((reference) => reference.stepEnd).filter(validStep);
    if (!starts.length || !ends.length) return null;
    return {
      stepStart: Math.min(...starts),
      stepEnd: Math.max(...ends),
    };
  }

  function curveColor(index) {
    return CURVE_COLORS[index % CURVE_COLORS.length];
  }

  function curveLabel(reference) {
    return `${referenceHeading(reference)} · steps ${reference.stepStart}–${reference.stepEnd}`;
  }

  function focusedExplorerUrl(reference) {
    const params = new URLSearchParams({
      run: reference.runId,
      spec: reference.specId,
      focus: "1",
      stepStart: String(reference.stepStart),
      stepEnd: String(reference.stepEnd),
    });
    if (
      typeof CuratorUI !== "undefined" &&
      typeof CuratorUI.isAdminEntry === "function" &&
      CuratorUI.isAdminEntry()
    ) {
      params.set("admin", "1");
    }
    return `explorer.html?${params.toString()}`;
  }

  async function loadReferencePoints(reference) {
    const manifest = await loadManifest(reference.runId);
    const spec = manifest._specMap.get(reference.specId);
    if (!spec) throw new Error("This observable is no longer available.");

    // Keep the final finite value for a repeated step, then sort it. This avoids
    // backward segments and ambiguous hover targets in older manifests.
    const byStep = new Map();
    const steps = spec.series.steps || [];
    const values = spec.series.values || [];
    const count = Math.min(steps.length, values.length);
    for (let index = 0; index < count; index += 1) {
      if (validStep(steps[index]) && Number.isFinite(values[index])) {
        byStep.set(steps[index], values[index]);
      }
    }
    const points = [...byStep]
      .sort(([left], [right]) => left - right)
      .map(([x, y]) => ({ x, y }))
      .filter(
        (point) =>
          point.x >= reference.stepStart && point.x <= reference.stepEnd
      );
    if (points.length < 2) {
      throw new Error("This region no longer contains enough recorded points.");
    }
    return points;
  }

  function setCurveKeyError(canvas, index, message) {
    const item = canvas
      .closest(".discovery-curve-group")
      ?.querySelector(`[data-curve-key-index="${index}"]`);
    if (!item) return;
    item.classList.add("is-error");
    const status = item.querySelector(".discovery-curve-state");
    if (status) status.textContent = message;
  }

  async function renderCombinedCurveChart(canvas, references, chartSet) {
    try {
      const requestedReferences = Array.isArray(references) ? references : [];
      const loaded = await Promise.all(
        requestedReferences.map(async (reference, index) => {
          try {
            return {
              index,
              reference,
              points: await loadReferencePoints(reference),
            };
          } catch (err) {
            return {
              index,
              reference,
              error: err?.message || String(err),
            };
          }
        })
      );

      // A refresh or removal may have replaced this canvas while data loaded.
      if (!canvas.isConnected) return;

      loaded.forEach((item) => {
        if (item.error) setCurveKeyError(canvas, item.index, item.error);
      });
      const available = loaded.filter((item) => item.points);
      if (!available.length) {
        throw new Error(
          requestedReferences.length
            ? "None of the selected curves could be loaded."
            : "No curves were selected."
        );
      }
      if (typeof Chart !== "function") {
        throw new Error("The chart library could not be loaded.");
      }

      const bounds = referenceBounds(requestedReferences);
      if (!bounds || bounds.stepEnd <= bounds.stepStart) {
        throw new Error("The selected curve range is invalid.");
      }
      const onlyOne = available.length === 1;
      const datasets = available.map(({ index, reference, points }) => {
        const color = curveColor(index);
        return {
          label: curveLabel(reference),
          data: points,
          borderColor: color,
          backgroundColor: `${color}20`,
          borderWidth: onlyOne ? 2.5 : 2.15,
          borderDash: CURVE_DASHES[index % CURVE_DASHES.length],
          pointRadius: 0,
          pointHoverRadius: 3.5,
          pointHitRadius: 8,
          cubicInterpolationMode: "monotone",
          tension: 0.18,
          spanGaps: false,
          fill: onlyOne,
        };
      });

      const chart = new Chart(canvas, {
        type: "line",
        data: { datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          parsing: false,
          // Curves may use different recording cadences and x positions.
          normalized: false,
          interaction: { intersect: false, mode: "nearest", axis: "x" },
          plugins: {
            // The HTML key stays readable with six long run/module/spec labels.
            legend: { display: false },
            tooltip: {
              callbacks: {
                title: (items) =>
                  items.length ? `step ${Math.round(items[0].parsed.x)}` : "",
              },
            },
          },
          scales: {
            x: {
              type: "linear",
              min: bounds.stepStart,
              max: bounds.stepEnd,
              grid: { color: "rgba(255,255,255,0.08)" },
              ticks: { color: "rgba(255,255,255,0.7)", maxTicksLimit: 6 },
              title: {
                display: true,
                text: `steps ${bounds.stepStart}–${bounds.stepEnd}`,
                color: "rgba(255,255,255,0.7)",
              },
            },
            y: {
              grid: { color: "rgba(255,255,255,0.08)" },
              ticks: { color: "rgba(255,255,255,0.7)", maxTicksLimit: 5 },
            },
          },
        },
      });
      chartSet.add(chart);
    } catch (err) {
      if (!canvas.isConnected) return;
      const host = canvas.closest(".discovery-curve-canvas");
      if (host) {
        host.innerHTML = `<p class="discovery-chart-error">${escapeHtml(
          err.message || String(err)
        )}</p>`;
      }
    }
  }

  function renderCurveKeyItem(reference, index, { removable = false } = {}) {
    const heading = referenceHeading(reference);
    return `
      <div class="discovery-curve-key-item" data-curve-key-index="${index}" role="listitem">
        <span
          class="discovery-curve-swatch"
          style="--curve-color: ${curveColor(index)}"
          aria-hidden="true"
        ></span>
        <div class="discovery-curve-key-copy">
          <strong>${escapeHtml(heading)}</strong>
          <span>steps ${reference.stepStart}–${reference.stepEnd}</span>
          <span class="discovery-curve-state" role="status"></span>
        </div>
        ${
          removable
            ? `<button type="button" class="note-delete" data-remove-reference="${index}" aria-label="Remove ${escapeHtml(heading)}">✕</button>`
            : `<a class="header-link" href="${escapeHtml(focusedExplorerUrl(reference))}">Open curve</a>`
        }
      </div>`;
  }

  function combinedRangeText(references) {
    const bounds = referenceBounds(references);
    if (!bounds) return "";
    return `steps ${bounds.stepStart}–${bounds.stepEnd}`;
  }

  function renderDraftReferences() {
    const host = document.getElementById("discoveryDraftReferences");
    destroyCharts(draftCharts);
    if (!draftReferences.length) {
      host.innerHTML = "";
      return;
    }
    host.innerHTML = `
      <article class="discovery-draft-reference discovery-curve-group">
        <div class="discovery-reference-head">
          <div>
            <strong>Combined curve preview</strong>
            <span>${draftReferences.length} ${draftReferences.length === 1 ? "curve" : "curves"} · ${combinedRangeText(draftReferences)}</span>
          </div>
        </div>
        <div class="discovery-curve-key" role="list" aria-label="Selected curves">
          ${draftReferences
            .map((reference, index) =>
              renderCurveKeyItem(reference, index, { removable: true })
            )
            .join("")}
        </div>
        <div class="discovery-curve-canvas">
          <canvas data-draft-chart aria-label="Combined preview of ${draftReferences.length} selected ${draftReferences.length === 1 ? "curve" : "curves"}"></canvas>
        </div>
      </article>`;
    host.querySelectorAll("[data-remove-reference]").forEach((button) => {
      button.addEventListener("click", () => {
        draftReferences.splice(Number(button.dataset.removeReference), 1);
        renderDraftReferences();
      });
    });
    const canvas = host.querySelector("[data-draft-chart]");
    if (canvas) renderCombinedCurveChart(canvas, draftReferences, draftCharts);
  }

  function buildCommentTree(comments, discoveryId) {
    const list = Array.isArray(comments) ? comments : [];
    const known = new Set(list.map((comment) => comment.id));
    const byParent = new Map();
    const rendered = new Set();
    for (const comment of list) {
      const parent =
        comment.parentId && known.has(comment.parentId) ? comment.parentId : "";
      if (!byParent.has(parent)) byParent.set(parent, []);
      byParent.get(parent).push(comment);
    }
    function level(parentId, depth, ancestors = new Set()) {
      if (depth > 20) return "";
      return (byParent.get(parentId || "") || [])
        .map((comment) => {
          if (rendered.has(comment.id) || ancestors.has(comment.id)) return "";
          rendered.add(comment.id);
          const nextAncestors = new Set(ancestors);
          nextAncestors.add(comment.id);
          return `
            <div class="note-comment discovery-comment ${depth ? "note-comment-reply" : ""}" data-comment-id="${escapeHtml(comment.id)}">
              <div class="note-comment-head">
                <div class="note-comment-byline">
                  ${comment.name ? `<span class="note-author">${escapeHtml(comment.name)}</span>` : `<span class="discovery-anonymous">Anonymous</span>`}
                  <time>${escapeHtml(formatTime(comment.createdAt))}</time>
                </div>
                <div class="note-comment-actions">
                  <button type="button" class="note-reply-btn" data-reply-discovery="${escapeHtml(discoveryId)}" data-parent-id="${escapeHtml(comment.id)}">Reply</button>
                  ${isAdmin ? `<button type="button" class="note-delete" data-delete-comment="${escapeHtml(comment.id)}" data-discovery-id="${escapeHtml(discoveryId)}" aria-label="Delete reply">✕</button>` : ""}
                </div>
              </div>
              <p>${escapeHtml(comment.text)}</p>
              ${level(comment.id, depth + 1, nextAncestors)}
            </div>`;
        })
        .join("");
    }
    let html = level("", 0);
    for (const comment of list) {
      if (rendered.has(comment.id)) continue;
      if (!byParent.has("")) byParent.set("", []);
      byParent.get("").push(comment);
      html += level("", 0);
    }
    return html;
  }

  function renderDiscoveryReferences(references, discoveryIndex) {
    return `
      <figure class="discovery-curve-reference discovery-curve-group">
        <figcaption>
          <div>
            <strong>Combined curve evidence</strong>
            <span>${references.length} ${references.length === 1 ? "curve" : "curves"} · ${combinedRangeText(references)}</span>
          </div>
        </figcaption>
        <div class="discovery-curve-key" role="list" aria-label="Curves in this chart">
          ${references
            .map((reference, index) => renderCurveKeyItem(reference, index))
            .join("")}
        </div>
        <div class="discovery-curve-canvas">
          <canvas
            data-feed-discovery="${discoveryIndex}"
            aria-label="Combined chart of ${references.length} ${references.length === 1 ? "curve" : "curves"}"
          ></canvas>
        </div>
      </figure>`;
  }

  function renderDiscoveries() {
    const host = document.getElementById("discoveryList");
    const status = document.getElementById("discoveryFeedStatus");
    destroyCharts(feedCharts);
    status.classList.remove("is-error");
    if (!discoveries.length) {
      host.innerHTML = "";
      status.textContent = "No discoveries yet. Be the first to document one.";
      return;
    }
    status.textContent = `${discoveries.length} ${discoveries.length === 1 ? "discovery" : "discoveries"}`;
    host.innerHTML = discoveries
      .map((discovery, discoveryIndex) => {
        const comments = buildCommentTree(discovery.comments, discovery.id);
        return `
          <article class="panel discovery-card" data-discovery-card="${escapeHtml(discovery.id)}">
            <header class="discovery-card-head">
              <div>
                <div class="discovery-card-meta">
                  <span class="note-author">${escapeHtml(discovery.name || "Anonymous")}</span>
                  <time>${escapeHtml(formatTime(discovery.createdAt))}</time>
                </div>
                <h3>${escapeHtml(discovery.title)}</h3>
              </div>
              ${isAdmin ? `<button type="button" class="chart-btn discovery-admin-delete" data-delete-discovery="${escapeHtml(discovery.id)}">Delete</button>` : ""}
            </header>
            <div class="discovery-body">${escapeHtml(discovery.body)}</div>
            ${
              discovery.references?.length
                ? `<div class="discovery-reference-list">${renderDiscoveryReferences(
                    discovery.references,
                    discoveryIndex
                  )}</div>`
                : ""
            }
            <section class="discovery-discussion" aria-label="Replies">
              <div class="discovery-discussion-head">
                <strong>Discussion</strong>
                <span>${(discovery.comments || []).length} ${(discovery.comments || []).length === 1 ? "reply" : "replies"}</span>
              </div>
              <div class="note-comments">${comments}</div>
              <button type="button" class="note-reply-btn note-reply-root" data-reply-discovery="${escapeHtml(discovery.id)}" data-parent-id="">Reply</button>
            </section>
          </article>`;
      })
      .join("");

    host.querySelectorAll("[data-feed-discovery]").forEach((canvas) => {
      const discovery = discoveries[Number(canvas.dataset.feedDiscovery)];
      const references = discovery?.references || [];
      if (references.length) {
        renderCombinedCurveChart(canvas, references, feedCharts);
      }
    });
    host.querySelectorAll("[data-reply-discovery]").forEach((button) => {
      button.addEventListener("click", () =>
        openReplyBox(
          button.dataset.replyDiscovery,
          button.dataset.parentId || null,
          button
        )
      );
    });
    host.querySelectorAll("[data-delete-discovery]").forEach((button) => {
      button.addEventListener("click", async () => {
        if (!confirm("Delete this discovery and every reply?")) return;
        button.disabled = true;
        try {
          await NotesStore.deleteDiscovery(button.dataset.deleteDiscovery);
          discoveries = discoveries.filter(
            (item) => item.id !== button.dataset.deleteDiscovery
          );
          renderDiscoveries();
        } catch (err) {
          alert(err.message || String(err));
          button.disabled = false;
        }
      });
    });
    host.querySelectorAll("[data-delete-comment]").forEach((button) => {
      button.addEventListener("click", async () => {
        if (!confirm("Delete this reply and its nested replies?")) return;
        try {
          await NotesStore.deleteDiscoveryComment(
            button.dataset.discoveryId,
            button.dataset.deleteComment
          );
          await refreshDiscoveries();
        } catch (err) {
          alert(err.message || String(err));
        }
      });
    });
  }

  function openReplyBox(discoveryId, parentId, anchor) {
    document.querySelectorAll(".discovery-reply-box").forEach((box) => box.remove());
    const form = document.createElement("form");
    form.className = "note-reply-box discovery-reply-box";
    form.innerHTML = `
      <input type="text" maxlength="80" autocomplete="name" placeholder="Leave your name (optional)" aria-label="Leave your name (optional)">
      <textarea rows="3" maxlength="2000" placeholder="Write a reply…" required></textarea>
      <div class="note-reply-box-actions">
        <button type="button" class="chart-btn discovery-reply-cancel">Cancel</button>
        <button type="submit" class="chart-btn note-submit">Post reply</button>
      </div>
      <p class="note-reply-status" hidden></p>`;
    const target =
      anchor.closest(".discovery-comment") ||
      anchor.closest(".discovery-discussion") ||
      anchor.closest(".discovery-card");
    target?.appendChild(form);
    const name = form.querySelector("input");
    const body = form.querySelector("textarea");
    const status = form.querySelector(".note-reply-status");
    body.focus();
    form.querySelector(".discovery-reply-cancel").addEventListener("click", () =>
      form.remove()
    );
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const text = body.value.trim();
      if (!text) return;
      const submit = form.querySelector("[type=submit]");
      submit.disabled = true;
      status.hidden = false;
      status.textContent = "Posting…";
      try {
        const comment = await NotesStore.createDiscoveryComment({
          discoveryId,
          parentId,
          name: name.value,
          text,
        });
        const discovery = discoveries.find((item) => item.id === discoveryId);
        if (discovery) {
          discovery.comments = discovery.comments || [];
          discovery.comments.push(comment);
        }
        renderDiscoveries();
      } catch (err) {
        status.textContent = err.message || String(err);
        submit.disabled = false;
      }
    });
  }

  async function refreshDiscoveries() {
    const status = document.getElementById("discoveryFeedStatus");
    const button = document.getElementById("discoveryRefresh");
    button.disabled = true;
    status.classList.remove("is-error");
    status.textContent = "Loading discoveries…";
    try {
      discoveries = await NotesStore.listDiscoveries();
      renderDiscoveries();
    } catch (err) {
      status.textContent = err.message || String(err);
      status.classList.add("is-error");
    } finally {
      button.disabled = false;
    }
  }

  async function publishDiscovery(event) {
    event.preventDefault();
    const button = document.getElementById("discoveryPublish");
    button.disabled = true;
    setFormStatus("Publishing…");
    try {
      const discovery = await NotesStore.createDiscovery({
        title: document.getElementById("discoveryTitle").value,
        body: document.getElementById("discoveryBody").value,
        name: document.getElementById("discoveryName").value,
        references: draftReferences,
      });
      discoveries.unshift(discovery);
      document.getElementById("discoveryForm").reset();
      draftReferences = [];
      renderDraftReferences();
      await populateRuns();
      renderDiscoveries();
      setFormStatus("Published.");
      document
        .getElementById("discoveryFeedTitle")
        .scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      setFormStatus(err.message || String(err), true);
    } finally {
      button.disabled = false;
    }
  }

  function syncAdminLinks() {
    if (typeof CuratorUI === "undefined" || !CuratorUI.withAdminParam) return;
    document
      .querySelectorAll("#discoveryHeaderLinks a.header-link")
      .forEach((link) => {
        const href = link.getAttribute("href");
        if (href) link.setAttribute("href", CuratorUI.withAdminParam(href));
      });
  }

  async function wire() {
    if (typeof MaintenanceGate !== "undefined") {
      const allowed = await MaintenanceGate.wire();
      if (!allowed) return;
    }
    if (typeof CuratorUI !== "undefined") CuratorUI.wire();
    else if (typeof AnnouncementBanner !== "undefined") AnnouncementBanner.wire();
    isAdmin =
      typeof CuratorUI !== "undefined" &&
      typeof CuratorUI.isAdmin === "function" &&
      CuratorUI.isAdmin();
    syncAdminLinks();
    document.addEventListener("curator-auth", (event) => {
      isAdmin = !!event.detail?.isAdmin;
      syncAdminLinks();
      renderDiscoveries();
    });

    document
      .getElementById("discoveryRun")
      .addEventListener("change", populateModules);
    document
      .getElementById("discoveryModule")
      .addEventListener("change", populateSpecs);
    document
      .getElementById("discoverySpec")
      .addEventListener("change", updateStepBounds);
    document
      .getElementById("discoveryAddReference")
      .addEventListener("click", () =>
        addDraftReference().catch((err) =>
          setReferenceStatus(err.message || String(err), true)
        )
      );
    document
      .getElementById("discoveryForm")
      .addEventListener("submit", publishDiscovery);
    document
      .getElementById("discoveryRefresh")
      .addEventListener("click", refreshDiscoveries);

    await NotesStore.init();
    await Promise.all([
      populateRuns().catch((err) =>
        setReferenceStatus(err.message || String(err), true)
      ),
      refreshDiscoveries(),
    ]);
  }

  return { wire };
})();

DiscoveriesUI.wire();
