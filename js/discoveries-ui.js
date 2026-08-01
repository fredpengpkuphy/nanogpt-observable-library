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
  const learningRateCache = new Map();
  const chartStateByCanvas = new WeakMap();
  const draftCharts = new Set();
  const feedCharts = new Set();
  let runs = [];
  let draftReferences = [];
  let discoveries = [];
  let isAdmin = false;
  let expandedCurveGroup = null;

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

  async function fetchText(url) {
    const separator = url.includes("?") ? "&" : "?";
    const response = await fetch(`${url}${separator}t=${Date.now()}`, {
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`Could not load ${url}`);
    return response.text();
  }

  function parseCsvRow(line) {
    const columns = [];
    let value = "";
    let quoted = false;
    for (let index = 0; index < line.length; index += 1) {
      const char = line[index];
      if (char === '"') {
        if (quoted && line[index + 1] === '"') {
          value += '"';
          index += 1;
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
  }

  function parseLearningRateCsv(text) {
    const lines = String(text || "").trim().split(/\r?\n/);
    if (lines.length < 2) return null;
    const headers = parseCsvRow(lines[0]).map((header, index) =>
      (index === 0 ? header.replace(/^\uFEFF/, "") : header).trim()
    );
    const stepIndex = headers.indexOf("iter");
    const lrIndex = headers.indexOf("lr");
    if (stepIndex < 0 || lrIndex < 0) return null;

    const byStep = new Map();
    for (let index = 1; index < lines.length; index += 1) {
      if (!lines[index].trim()) continue;
      const columns = parseCsvRow(lines[index]);
      const rawStep = columns[stepIndex];
      const rawRate = columns[lrIndex];
      if (
        rawStep == null ||
        rawRate == null ||
        String(rawStep).trim() === "" ||
        String(rawRate).trim() === ""
      ) {
        continue;
      }
      const step = Number(rawStep);
      const rate = Number(rawRate);
      if (validStep(step) && Number.isFinite(rate) && rate >= 0) {
        byStep.set(step, { x: step, y: rate });
      }
    }
    const lr = [...byStep.values()].sort((left, right) => left.x - right.x);
    return lr.length ? { lr } : null;
  }

  async function loadLearningRateLog(runId) {
    if (learningRateCache.has(runId)) return learningRateCache.get(runId);
    const promise = fetchText(
      `data/${encodeURIComponent(runId)}/eval_loss_log.csv`
    ).then((text) => {
      const parsed = parseLearningRateCsv(text);
      if (!parsed) throw new Error(`Learning-rate data is unavailable for ${runId}.`);
      return parsed;
    });
    learningRateCache.set(runId, promise);
    try {
      return await promise;
    } catch (err) {
      learningRateCache.delete(runId);
      throw err;
    }
  }

  function learningRateAtStep(log, step) {
    const points = log?.lr || [];
    if (!points.length || !Number.isFinite(step) || step < 0) return NaN;
    if (step <= points[0].x) return points[0].y;
    const last = points[points.length - 1];
    if (step >= last.x) return last.y;

    let low = 0;
    let high = points.length - 1;
    while (high - low > 1) {
      const middle = Math.floor((low + high) / 2);
      if (points[middle].x <= step) low = middle;
      else high = middle;
    }
    const left = points[low];
    const right = points[high];
    const span = right.x - left.x;
    if (!(span > 0)) return left.y;
    const fraction = (step - left.x) / span;
    return left.y + fraction * (right.y - left.y);
  }

  function ensureLearningRatePrefix(log) {
    if (log?._lrIntegral) return log._lrIntegral;
    const points = log?.lr || [];
    if (!points.length) return null;
    const prefixAtPoint = new Array(points.length).fill(0);
    prefixAtPoint[0] = points[0].x * points[0].y;
    for (let index = 0; index + 1 < points.length; index += 1) {
      const left = points[index];
      const right = points[index + 1];
      const count = right.x - left.x;
      const slope = count > 0 ? (right.y - left.y) / count : 0;
      const area = count * left.y + slope * count * (count - 1) / 2;
      prefixAtPoint[index + 1] = prefixAtPoint[index] + area;
      if (!Number.isFinite(prefixAtPoint[index + 1])) return null;
    }
    log._lrIntegral = { points, prefixAtPoint };
    return log._lrIntegral;
  }

  function cumulativeLearningRateAt(log, step) {
    if (!Number.isFinite(step) || step < 0) return NaN;
    if (step === 0) return 0;
    const integral = ensureLearningRatePrefix(log);
    if (!integral) return NaN;
    const { points, prefixAtPoint } = integral;
    const whole = Math.floor(step);
    let tau;
    if (whole <= points[0].x) {
      tau = whole * points[0].y;
    } else {
      let low = 0;
      let high = points.length - 1;
      while (low < high) {
        const middle = Math.ceil((low + high) / 2);
        if (points[middle].x <= whole) low = middle;
        else high = middle - 1;
      }
      const left = points[low];
      const count = whole - left.x;
      tau = prefixAtPoint[low];
      if (count > 0) {
        if (low + 1 < points.length) {
          const right = points[low + 1];
          const span = right.x - left.x;
          const slope = span > 0 ? (right.y - left.y) / span : 0;
          tau += count * left.y + slope * count * (count - 1) / 2;
        } else {
          tau += count * left.y;
        }
      }
    }
    const fraction = step - whole;
    if (fraction > 0) tau += fraction * learningRateAtStep(log, whole);
    return Number.isFinite(tau) && tau >= 0 ? tau : NaN;
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
      const canvas = chart?.canvas;
      if (canvas?.closest(".discovery-curve-group") === expandedCurveGroup) {
        closeExpandedCurveGroup();
      }
      if (canvas) {
        canvas.ondblclick = null;
        chartStateByCanvas.delete(canvas);
      }
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

  function setCurveViewStatus(group, message = "", isError = false) {
    const status = group?.querySelector(".discovery-curve-view-status");
    if (!status) return;
    status.textContent = message;
    status.classList.toggle("is-error", !!isError);
  }

  function stateForGroup(group) {
    const canvas = group?.querySelector("canvas[data-discovery-curve-chart]");
    return canvas ? chartStateByCanvas.get(canvas) : null;
  }

  function setCurveControlsReady(group, state = null) {
    group?.querySelectorAll("[data-curve-view-control]").forEach((control) => {
      control.disabled = !state;
    });
    group?.querySelectorAll("[data-toggle-curve]").forEach((button) => {
      const index = Number(button.dataset.toggleCurve);
      button.disabled =
        !state || !state.definitions.some((definition) => definition.index === index);
    });
  }

  function scaleUsesLogY(scaleMode) {
    return scaleMode === "loglinear" || scaleMode === "loglog";
  }

  function formatAxisValue(value) {
    if (!Number.isFinite(value)) return "";
    const magnitude = Math.abs(value);
    if ((magnitude > 0 && magnitude < 1e-3) || magnitude >= 1e6) {
      return value.toExponential(3);
    }
    return value.toLocaleString(undefined, { maximumSignificantDigits: 6 });
  }

  function axisValueForStep(state, reference, step, axisMode) {
    if (axisMode === "step") return step;
    return cumulativeLearningRateAt(
      state.learningRateByRun.get(reference.runId),
      step
    );
  }

  function buildCurveDatasets(state, scaleMode, axisMode) {
    const logX = scaleMode === "loglog";
    const mapped = state.definitions.map((definition) => ({
      definition,
      points: definition.rawPoints
        .map((point) => ({
          axisX: axisValueForStep(
            state,
            definition.reference,
            point.x,
            axisMode
          ),
          y: point.y,
          step: point.x,
        }))
        .filter((point) => Number.isFinite(point.axisX) && point.axisX >= 0),
    }));
    const axisValues = mapped.flatMap((item) =>
      item.points.map((point) => point.axisX)
    );
    if (!axisValues.length) {
      throw new Error(
        axisMode === "tau"
          ? "No curves can be mapped to τ."
          : "No valid horizontal-axis values are available."
      );
    }
    const rangeValues =
      axisMode === "step" && state.bounds
        ? [state.bounds.stepStart, state.bounds.stepEnd]
        : axisValues;
    const positiveValues = [...axisValues, ...rangeValues].filter(
      (value) => value > 0
    );
    if (logX && !positiveValues.length) {
      throw new Error("Log–log view requires at least one positive x value.");
    }
    const zeroPlotX = positiveValues.length
      ? Math.min(...positiveValues) / 10
      : Number.EPSILON;
    const plottedValues = rangeValues.map((value) =>
      logX && value === 0 ? zeroPlotX : value
    );
    let xMin = Math.min(...plottedValues);
    let xMax = Math.max(...plottedValues);
    if (!(xMax > xMin)) {
      if (logX) {
        xMin = Math.max(xMin / 10, Number.EPSILON);
        xMax = Math.max(xMax * 10, xMin * 10);
      } else {
        const span = axisMode === "tau" ? 1e-12 : 1;
        xMin = Math.max(0, xMin - span / 2);
        xMax = xMin + span;
      }
    }

    const visibleCount = state.definitions.filter(
      (definition) => !definition.hidden
    ).length;
    return {
      xMin,
      xMax,
      datasets: mapped.map(({ definition, points }) => {
        const displayPoints = scaleUsesLogY(scaleMode)
          ? points.filter((point) => point.y > 0)
          : points;
        return {
          label: curveLabel(definition.reference),
          data: displayPoints.map((point) => ({
            x: logX && point.axisX === 0 ? zeroPlotX : point.axisX,
            y: point.y,
            _step: point.step,
            ...(axisMode === "tau" ? { _tau: point.axisX } : {}),
          })),
          borderColor: curveColor(definition.index),
          backgroundColor: `${curveColor(definition.index)}20`,
          borderWidth: visibleCount === 1 ? 2.5 : 2.15,
          borderDash: CURVE_DASHES[definition.index % CURVE_DASHES.length],
          pointRadius: 0,
          pointHoverRadius: 3.5,
          pointHitRadius: 8,
          cubicInterpolationMode: "monotone",
          tension: 0.18,
          spanGaps: false,
          fill: visibleCount === 1 && !scaleUsesLogY(scaleMode),
          hidden: definition.hidden,
          _referenceIndex: definition.index,
        };
      }),
    };
  }

  function curveChartOptions(state, xMin, xMax) {
    const logX = state.scaleMode === "loglog";
    const logY = scaleUsesLogY(state.scaleMode);
    const axisColor = "rgba(255,255,255,0.7)";
    return {
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
            title: (items) => {
              if (!items.length) return "";
              const raw = items[0].raw;
              if (state.axisMode === "tau" && Number.isFinite(raw?._tau)) {
                return `τ ${formatAxisValue(raw._tau)} · step ${raw._step}`;
              }
              return Number.isFinite(raw?._step) ? `step ${raw._step}` : "";
            },
          },
        },
        zoom: {
          zoom: {
            wheel: { enabled: true, speed: 0.1 },
            pinch: { enabled: true },
            mode: "xy",
            drag: {
              enabled: !!state.expanded,
              threshold: 8,
              backgroundColor: "rgba(159, 212, 228, 0.14)",
              borderColor: "rgba(159, 212, 228, 0.85)",
              borderWidth: 1,
            },
          },
          pan: {
            enabled: true,
            mode: "xy",
            modifierKey: "alt",
            threshold: 10,
          },
          limits: {
            x: {
              min: "original",
              max: "original",
              minRange: state.axisMode === "tau" ? 1e-12 : 1,
            },
            y: { min: "original", max: "original", minRange: 1e-12 },
          },
        },
      },
      scales: {
        x: {
          type: logX ? "logarithmic" : "linear",
          min: xMin,
          max: xMax,
          grid: { color: "rgba(255,255,255,0.08)" },
          ticks: { color: axisColor, maxTicksLimit: 6 },
          title: {
            display: true,
            text: `${state.axisMode === "tau" ? "τ" : "steps"}${logX ? " (log)" : ""}`,
            color: axisColor,
          },
        },
        y: {
          type: logY ? "logarithmic" : "linear",
          grid: { color: "rgba(255,255,255,0.08)" },
          ticks: { color: axisColor, maxTicksLimit: 5 },
          title: {
            display: true,
            text: logY ? "Value (log)" : "Value",
            color: axisColor,
          },
        },
      },
    };
  }

  function syncCurveViewControls(state) {
    const group = state.group;
    group.querySelectorAll("[data-curve-scale]").forEach((button) => {
      button.classList.toggle("active", button.dataset.curveScale === state.scaleMode);
    });
    group.querySelectorAll("[data-curve-axis]").forEach((button) => {
      button.classList.toggle("active", button.dataset.curveAxis === state.axisMode);
    });
    for (const definition of state.definitions) {
      const item = group.querySelector(
        `[data-curve-key-index="${definition.index}"]`
      );
      const button = item?.querySelector("[data-toggle-curve]");
      item?.classList.toggle("is-hidden", definition.hidden);
      if (button) {
        button.textContent = definition.hidden ? "Show" : "Hide";
        button.setAttribute("aria-pressed", String(!definition.hidden));
        button.setAttribute(
          "aria-label",
          `${definition.hidden ? "Show" : "Hide"} ${referenceHeading(
            definition.reference
          )}`
        );
      }
    }
    const expand = group.querySelector('[data-curve-action="expand"]');
    if (expand) {
      expand.textContent = state.expanded ? "✕ Close" : "⤢ Expand";
      expand.setAttribute(
        "aria-label",
        state.expanded ? "Close expanded chart" : "Expand chart"
      );
    }
  }

  function mountCurveChart(state) {
    const built = buildCurveDatasets(state, state.scaleMode, state.axisMode);
    const previous = state.chart;
    if (previous) {
      state.chartSet.delete(previous);
      try {
        previous.destroy();
      } catch (_) {}
    }
    state.canvas.removeAttribute("width");
    state.canvas.removeAttribute("height");
    state.canvas.style.width = "";
    state.canvas.style.height = "";
    const chart = new Chart(state.canvas, {
      type: "line",
      data: { datasets: built.datasets },
      options: curveChartOptions(state, built.xMin, built.xMax),
    });
    state.chart = chart;
    state.chartSet.add(chart);
    chartStateByCanvas.set(state.canvas, state);
    state.canvas.ondblclick = (event) => {
      event.preventDefault();
      resetCurveChart(state);
    };
    setCurveControlsReady(state.group, state);
    syncCurveViewControls(state);
    return chart;
  }

  function replaceCurveChartView(state, { scaleMode, axisMode } = {}) {
    const nextScale = scaleMode || state.scaleMode;
    const nextAxis = axisMode || state.axisMode;
    if (scaleUsesLogY(nextScale)) {
      const blocker = state.definitions.find(
        (definition) =>
          !definition.hidden &&
          definition.rawPoints.some(
            (point) => Number.isFinite(point.y) && point.y <= 0
          )
      );
      if (blocker) {
        setCurveViewStatus(
          state.group,
          `Logarithmic y-axis unavailable: ${referenceHeading(
            blocker.reference
          )} contains zero or negative values.`,
          true
        );
        return false;
      }
    }

    const previousScale = state.scaleMode;
    const previousAxis = state.axisMode;
    state.scaleMode = nextScale;
    state.axisMode = nextAxis;
    try {
      mountCurveChart(state);
      const visible = state.definitions.filter(
        (definition) => !definition.hidden
      ).length;
      setCurveViewStatus(
        state.group,
        visible ? "" : "All curves are hidden. Use Show to restore one."
      );
      return true;
    } catch (err) {
      state.scaleMode = previousScale;
      state.axisMode = previousAxis;
      try {
        mountCurveChart(state);
      } catch (_) {
        state.chart = null;
        setCurveControlsReady(state.group, null);
      }
      setCurveViewStatus(state.group, err?.message || String(err), true);
      return false;
    }
  }

  async function setCurveAxisMode(state, axisMode) {
    if (!state || !["step", "tau"].includes(axisMode)) return;
    const request = ++state.axisRequest;
    if (axisMode === "step") {
      replaceCurveChartView(state, { axisMode });
      return;
    }
    setCurveViewStatus(state.group, "Loading learning-rate data…");
    try {
      const runIds = [...new Set(
        state.definitions.map((definition) => definition.reference.runId)
      )];
      const logs = await Promise.all(
        runIds.map(async (runId) => [runId, await loadLearningRateLog(runId)])
      );
      if (
        request !== state.axisRequest ||
        !state.canvas.isConnected ||
        chartStateByCanvas.get(state.canvas) !== state
      ) {
        return;
      }
      state.learningRateByRun = new Map(logs);
      const hasPositiveTau = state.definitions.some((definition) =>
        definition.rawPoints.some(
          (point) =>
            cumulativeLearningRateAt(
              state.learningRateByRun.get(definition.reference.runId),
              point.x
            ) > 0
        )
      );
      if (!hasPositiveTau) {
        throw new Error("τ view requires a positive recorded learning rate.");
      }
      replaceCurveChartView(state, { axisMode });
    } catch (err) {
      if (request !== state.axisRequest) return;
      setCurveViewStatus(state.group, err?.message || String(err), true);
      syncCurveViewControls(state);
    }
  }

  function fallbackZoomScale(scale, magnification) {
    const min = Number(scale?.min);
    const max = Number(scale?.max);
    if (!Number.isFinite(min) || !Number.isFinite(max) || !(max > min)) return null;
    if (scale.type === "logarithmic") {
      if (!(min > 0)) return null;
      const logMin = Math.log(min);
      const logMax = Math.log(max);
      const center = (logMin + logMax) / 2;
      const halfSpan = (logMax - logMin) / (2 * magnification);
      return {
        min: Math.exp(center - halfSpan),
        max: Math.exp(center + halfSpan),
      };
    }
    const center = min + (max - min) / 2;
    const span = (max - min) / magnification;
    let nextMin = center - span / 2;
    let nextMax = center + span / 2;
    if (scale.axis === "x" && nextMin < 0) {
      nextMax -= nextMin;
      nextMin = 0;
    }
    return { min: nextMin, max: nextMax };
  }

  function zoomCurveChart(state, magnification) {
    const chart = state?.chart;
    if (!chart || !(magnification > 0) || magnification === 1) return;
    try {
      if (typeof chart.zoom === "function") {
        chart.zoom({ x: magnification, y: magnification }, "none");
      } else {
        for (const axis of ["x", "y"]) {
          const bounds = fallbackZoomScale(chart.scales?.[axis], magnification);
          if (!bounds) continue;
          chart.options.scales[axis].min = bounds.min;
          chart.options.scales[axis].max = bounds.max;
        }
        chart.update("none");
      }
      setCurveViewStatus(state.group, "");
    } catch (err) {
      setCurveViewStatus(state.group, "Chart zoom failed.", true);
      console.warn("Discovery chart zoom failed", err);
    }
  }

  function resetCurveChart(state) {
    if (!state?.chart) return;
    replaceCurveChartView(state);
  }

  function scheduleCurveResize(state) {
    const resize = () => {
      try {
        state?.chart?.resize();
      } catch (_) {}
    };
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(resize);
    else setTimeout(resize, 0);
  }

  function closeExpandedCurveGroup() {
    const group = expandedCurveGroup;
    if (!group) return;
    const state = stateForGroup(group);
    group.classList.remove("is-expanded");
    group.removeAttribute("role");
    group.removeAttribute("aria-modal");
    document.body.classList.remove("discovery-chart-expanded");
    expandedCurveGroup = null;
    if (state) {
      state.expanded = false;
      const drag = state.chart?.options?.plugins?.zoom?.zoom?.drag;
      if (drag) drag.enabled = false;
      try {
        state.chart?.update?.("none");
      } catch (_) {}
      syncCurveViewControls(state);
      scheduleCurveResize(state);
    }
  }

  function toggleExpandedCurveGroup(state) {
    if (!state?.group) return;
    if (expandedCurveGroup === state.group) {
      closeExpandedCurveGroup();
      return;
    }
    closeExpandedCurveGroup();
    expandedCurveGroup = state.group;
    state.expanded = true;
    state.group.classList.add("is-expanded");
    state.group.setAttribute("role", "dialog");
    state.group.setAttribute("aria-modal", "true");
    document.body.classList.add("discovery-chart-expanded");
    const drag = state.chart?.options?.plugins?.zoom?.zoom?.drag;
    if (drag) drag.enabled = true;
    try {
      state.chart?.update?.("none");
    } catch (_) {}
    syncCurveViewControls(state);
    scheduleCurveResize(state);
  }

  function toggleCurveVisibility(state, referenceIndex) {
    const definition = state?.definitions.find(
      (candidate) => candidate.index === referenceIndex
    );
    if (!definition || !state.chart) return;
    if (
      definition.hidden &&
      scaleUsesLogY(state.scaleMode) &&
      definition.rawPoints.some(
        (point) => Number.isFinite(point.y) && point.y <= 0
      )
    ) {
      setCurveViewStatus(
        state.group,
        "This curve contains zero or negative values and cannot be shown on a logarithmic y-axis.",
        true
      );
      return;
    }
    definition.hidden = !definition.hidden;
    const datasetIndex = state.chart.data.datasets.findIndex(
      (dataset) => dataset._referenceIndex === referenceIndex
    );
    if (datasetIndex >= 0) {
      state.chart.setDatasetVisibility(datasetIndex, !definition.hidden);
      state.chart.update("none");
    }
    syncCurveViewControls(state);
    const visible = state.definitions.filter((item) => !item.hidden).length;
    setCurveViewStatus(
      state.group,
      visible ? "" : "All curves are hidden. Use Show to restore one."
    );
  }

  function wireCurveGroup(group) {
    group.querySelectorAll("[data-curve-scale]").forEach((button) => {
      button.addEventListener("click", () => {
        const state = stateForGroup(group);
        if (state) {
          replaceCurveChartView(state, { scaleMode: button.dataset.curveScale });
        }
      });
    });
    group.querySelectorAll("[data-curve-axis]").forEach((button) => {
      button.addEventListener("click", () => {
        const state = stateForGroup(group);
        if (state) setCurveAxisMode(state, button.dataset.curveAxis);
      });
    });
    group.querySelectorAll("[data-curve-action]").forEach((button) => {
      button.addEventListener("click", () => {
        const state = stateForGroup(group);
        if (!state) return;
        if (button.dataset.curveAction === "zoom-in") zoomCurveChart(state, 1.25);
        else if (button.dataset.curveAction === "zoom-out") zoomCurveChart(state, 0.8);
        else if (button.dataset.curveAction === "reset") resetCurveChart(state);
        else if (button.dataset.curveAction === "expand") {
          toggleExpandedCurveGroup(state);
        }
      });
    });
    group.querySelectorAll("[data-toggle-curve]").forEach((button) => {
      button.addEventListener("click", () => {
        const state = stateForGroup(group);
        if (state) toggleCurveVisibility(state, Number(button.dataset.toggleCurve));
      });
    });
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
      const group = canvas.closest(".discovery-curve-group");
      if (!group) throw new Error("The chart container is unavailable.");
      const state = {
        canvas,
        group,
        chartSet,
        chart: null,
        bounds,
        definitions: available.map(({ index, reference, points }) => ({
          index,
          reference,
          rawPoints: points,
          hidden: false,
        })),
        scaleMode: "linear",
        axisMode: "step",
        learningRateByRun: new Map(),
        axisRequest: 0,
        expanded: false,
      };
      mountCurveChart(state);
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

  function renderCurveToolbar() {
    return `
      <div class="discovery-curve-toolbar" aria-label="Curve view controls">
        <div class="loss-view-toggle" title="Axis scale">
          <button type="button" class="chart-btn curve-scale-btn active" data-curve-scale="linear" data-curve-view-control disabled>Linear</button>
          <button type="button" class="chart-btn curve-scale-btn" data-curve-scale="loglinear" data-curve-view-control disabled>Log–linear</button>
          <button type="button" class="chart-btn curve-scale-btn" data-curve-scale="loglog" data-curve-view-control disabled>Log–log</button>
        </div>
        <div class="loss-view-toggle discovery-axis-toggle" title="Horizontal axis">
          <button type="button" class="chart-btn discovery-axis-btn active" data-curve-axis="step" data-curve-view-control disabled>Step</button>
          <button type="button" class="chart-btn discovery-axis-btn" data-curve-axis="tau" data-curve-view-control disabled>τ</button>
        </div>
        <div class="chart-zoom-controls" aria-label="Curve zoom">
          <button type="button" class="chart-btn chart-zoom-btn" data-curve-action="zoom-in" data-curve-view-control title="Zoom in" aria-label="Zoom in" disabled>＋</button>
          <button type="button" class="chart-btn chart-zoom-btn" data-curve-action="zoom-out" data-curve-view-control title="Zoom out" aria-label="Zoom out" disabled>−</button>
        </div>
        <button type="button" class="chart-btn" data-curve-action="reset" data-curve-view-control title="Reset zoom and view" disabled>Reset</button>
        <button type="button" class="chart-btn discovery-expand-curve" data-curve-action="expand" data-curve-view-control title="Expand chart" disabled>⤢ Expand</button>
        <span class="discovery-curve-view-status" role="status"></span>
      </div>
      <p class="discovery-curve-hint">Scroll or pinch to zoom · Alt+drag to pan · double-click to reset · expanded view also supports drag-to-zoom</p>`;
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
        <div class="discovery-curve-key-actions">
          <button type="button" class="chart-btn discovery-curve-visibility" data-toggle-curve="${index}" data-curve-view-control aria-pressed="true" aria-label="Hide ${escapeHtml(heading)}" disabled>Hide</button>
          ${
            removable
              ? `<button type="button" class="note-delete" data-remove-reference="${index}" aria-label="Remove ${escapeHtml(heading)}">✕</button>`
              : `<a class="header-link" href="${escapeHtml(focusedExplorerUrl(reference))}">Open curve</a>`
          }
        </div>
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
        ${renderCurveToolbar()}
        <div class="discovery-curve-key" role="list" aria-label="Selected curves">
          ${draftReferences
            .map((reference, index) =>
              renderCurveKeyItem(reference, index, { removable: true })
            )
            .join("")}
        </div>
        <div class="discovery-curve-canvas">
          <canvas data-discovery-curve-chart data-draft-chart aria-label="Combined preview of ${draftReferences.length} selected ${draftReferences.length === 1 ? "curve" : "curves"}"></canvas>
        </div>
      </article>`;
    host.querySelectorAll("[data-remove-reference]").forEach((button) => {
      button.addEventListener("click", () => {
        draftReferences.splice(Number(button.dataset.removeReference), 1);
        renderDraftReferences();
      });
    });
    const group = host.querySelector(".discovery-curve-group");
    if (group) wireCurveGroup(group);
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
        ${renderCurveToolbar()}
        <div class="discovery-curve-key" role="list" aria-label="Curves in this chart">
          ${references
            .map((reference, index) => renderCurveKeyItem(reference, index))
            .join("")}
        </div>
        <div class="discovery-curve-canvas">
          <canvas
            data-discovery-curve-chart
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

    host.querySelectorAll(".discovery-curve-group").forEach(wireCurveGroup);

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
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && expandedCurveGroup) {
        event.preventDefault();
        closeExpandedCurveGroup();
      }
    });

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
