/**
 * Fullscreen chart annotations: click a step → leave a public note.
 */

let allNotes = [];
let notesReady = false;
let pendingNoteStep = null;
let noteClickTimer = null;
let noteClickBoundCanvas = null;

function noteContextKey() {
  if (fullOverlayMode === "loss") {
    return { runId, specId: "__loss__", context: "loss" };
  }
  return {
    runId,
    specId: activeSpecId || "",
    context: "spec",
  };
}

function notesForCurrentChart() {
  const key = noteContextKey();
  if (!key.runId || (!key.specId && key.context === "spec")) return [];
  return NotesStore.filterNotes(allNotes, key);
}

function chartIsAlive(chart) {
  try {
    return !!(chart && chart.canvas && chart.ctx);
  } catch (_) {
    return false;
  }
}

async function refreshNotes() {
  try {
    allNotes = await NotesStore.listNotes();
    notesReady = true;
  } catch (err) {
    console.warn(err);
    allNotes = [];
    notesReady = false;
  }
  updateNotesHint();
  if (fullOverlayOpen && chartIsAlive(fullChart)) {
    applyNoteAnnotations(fullChart);
  }
}

function updateNotesHint() {
  const hint = document.getElementById("fullChartHint");
  if (!hint) return;
  if (!fullOverlayOpen) return;
  const n = notesForCurrentChart().length;
  const base =
    "Click the curve to leave a note · or use Note · scroll to zoom · double-click to reset · Esc to close";
  hint.textContent = n ? `${base} · ${n} note${n === 1 ? "" : "s"} on this curve` : base;
}

function collectChartSteps(chart) {
  const steps = new Set();
  for (const ds of chart.data?.datasets || []) {
    for (const p of ds.data || []) {
      if (!p || !Number.isFinite(p.x)) continue;
      // Loss log–log plots at step+1 and stores the true step in `_step`.
      steps.add(p._step != null ? p._step : p.x);
    }
  }
  return [...steps].sort((a, b) => a - b);
}

function nearestStep(chart, pixelX) {
  const xScale = chart.scales?.x;
  if (!xScale) return null;
  const xVal = xScale.getValueForPixel(pixelX);
  if (!Number.isFinite(xVal)) return null;
  let best = null;
  let bestDist = Infinity;
  for (const ds of chart.data?.datasets || []) {
    for (const p of ds.data || []) {
      if (!p || !Number.isFinite(p.x)) continue;
      const d = Math.abs(p.x - xVal);
      if (d < bestDist) {
        bestDist = d;
        best = p._step != null ? p._step : p.x;
      }
    }
  }
  if (best != null) return best;
  return Math.round(xVal);
}

function plotXForNoteStep(step) {
  // Loss log–log charts plot at step+1 (log axis cannot show 0).
  if (
    typeof fullOverlayMode !== "undefined" &&
    fullOverlayMode === "loss" &&
    typeof lossScaleMode !== "undefined" &&
    lossScaleMode === "loglog"
  ) {
    return Number(step) + 1;
  }
  return step;
}

function noteAnnotationConfig(notes) {
  const byStep = new Map();
  for (const n of notes) {
    if (!byStep.has(n.step)) byStep.set(n.step, []);
    byStep.get(n.step).push(n);
  }
  const annotations = {};
  let i = 0;
  for (const [step, list] of byStep) {
    const x = plotXForNoteStep(step);
    annotations[`note_${i++}`] = {
      type: "line",
      xMin: x,
      xMax: x,
      borderColor: "rgba(232, 200, 136, 0.9)",
      borderWidth: 1.5,
      borderDash: [4, 3],
      label: {
        display: true,
        content: list.length > 1 ? `${list.length} notes` : "note",
        position: "start",
        backgroundColor: "rgba(14, 20, 25, 0.8)",
        color: "#ffe6b8",
        font: { size: 10, weight: "600" },
        padding: 3,
      },
    };
  }
  return annotations;
}

function applyNoteAnnotations(chart) {
  if (!chartIsAlive(chart)) return;
  const notes = notesForCurrentChart();
  try {
    if (!chart.options.plugins) chart.options.plugins = {};
    // Only touch annotation config — never replace the whole plugins object
    // (that would drop zoom/legend and can blank the chart after update).
    chart.options.plugins.annotation = {
      annotations: noteAnnotationConfig(notes),
    };
    chart.update("none");
  } catch (err) {
    console.warn("applyNoteAnnotations failed", err);
  }
  updateNotesHint();
  renderNotesRail(notes);
}

function renderNotesRail(notes) {
  const rail = document.getElementById("notesRail");
  if (!rail) return;
  if (!notes.length) {
    rail.innerHTML = `<p class="notes-rail-empty">No notes yet. Click the chart to add one.</p>`;
    return;
  }
  const sorted = [...notes].sort((a, b) => a.step - b.step || a.createdAt.localeCompare(b.createdAt));
  rail.innerHTML = sorted
    .map(
      (n) => `
    <article class="note-card" data-step="${n.step}">
      <header>
        <span class="note-step">step ${n.step}</span>
        <span class="note-author">${escapeHtml(n.author)}</span>
      </header>
      <p>${escapeHtml(n.text)}</p>
      <a class="note-issue" href="${n.issueUrl}" target="_blank" rel="noopener">GitHub #${n.issueNumber}</a>
    </article>`
    )
    .join("");
  rail.querySelectorAll(".note-card").forEach((card) => {
    card.addEventListener("click", () => {
      const step = Number(card.dataset.step);
      openNoteModal(step, { viewOnly: false });
    });
  });
}

function cancelPendingNoteClick() {
  if (noteClickTimer) {
    clearTimeout(noteClickTimer);
    noteClickTimer = null;
  }
}

let notePointerDown = null;
let noteCanvasClickBound = null;

function canvasCssPixelX(chart, clientX) {
  const canvas = chart?.canvas;
  if (!canvas) return null;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width) return null;
  // Match Chart.js getRelativePosition: CSS pixels in chart coordinate space.
  return ((clientX - rect.left) / rect.width) * chart.width;
}

function scheduleOpenNoteAtStep(step) {
  if (step == null || !Number.isFinite(step)) return;
  cancelPendingNoteClick();
  noteClickTimer = setTimeout(() => {
    noteClickTimer = null;
    if (!fullOverlayOpen || !chartIsAlive(fullChart)) return;
    if (typeof isNoteModalOpen === "function" && isNoteModalOpen()) return;
    openNoteModal(step);
  }, 280);
}

function onFullscreenCanvasPointerDown(evt) {
  if (evt.button != null && evt.button !== 0) return;
  notePointerDown = { x: evt.clientX, y: evt.clientY, t: Date.now() };
}

/**
 * Native canvas click — does NOT use Chart.js onClick.
 * Chart.js only invokes onClick when the pointer is inside chartArea, and
 * zoom/Hammer can swallow clicks; both made the note modal appear "broken".
 */
function onFullscreenCanvasClick(evt) {
  if (!fullOverlayOpen || !chartIsAlive(fullChart)) return;
  if (typeof isNoteModalOpen === "function" && isNoteModalOpen()) return;
  if (evt.detail > 1) {
    cancelPendingNoteClick();
    return;
  }
  if (notePointerDown) {
    const dx = evt.clientX - notePointerDown.x;
    const dy = evt.clientY - notePointerDown.y;
    if (Math.hypot(dx, dy) > 10) {
      notePointerDown = null;
      return;
    }
  }
  notePointerDown = null;
  const pixelX = canvasCssPixelX(fullChart, evt.clientX);
  if (pixelX == null) return;
  const step = nearestStep(fullChart, pixelX);
  if (step == null) return;
  scheduleOpenNoteAtStep(step);
}

/** Open note at midpoint step (toolbar fallback when click path fails). */
function openNoteModalAtChartCenter() {
  if (!chartIsAlive(fullChart)) return;
  const steps = collectChartSteps(fullChart);
  let step;
  if (steps.length) {
    step = steps[Math.floor(steps.length / 2)];
  } else {
    const xScale = fullChart.scales?.x;
    step = xScale ? Math.round((xScale.min + xScale.max) / 2) : 0;
  }
  openNoteModal(step);
}

function detachFullscreenNoteHandlers(chart) {
  cancelPendingNoteClick();
  notePointerDown = null;
  const canvas = noteCanvasClickBound || chart?.canvas || noteClickBoundCanvas;
  if (canvas && noteCanvasClickBound) {
    canvas.removeEventListener("pointerdown", onFullscreenCanvasPointerDown);
    canvas.removeEventListener("click", onFullscreenCanvasClick);
    canvas.style.cursor = "";
  }
  noteCanvasClickBound = null;
  noteClickBoundCanvas = null;
}

/** Kept for app.js Chart onClick wiring (secondary path). */
function handleFullscreenChartClick(evt, chart) {
  if (!fullOverlayOpen || !chartIsAlive(chart)) return;
  if (typeof isNoteModalOpen === "function" && isNoteModalOpen()) return;
  const native = evt?.native || evt;
  if (native?.detail > 1) {
    cancelPendingNoteClick();
    return;
  }
  let pixelX = null;
  if (typeof evt?.x === "number" && Number.isFinite(evt.x)) {
    pixelX = evt.x;
  } else if (native?.clientX != null) {
    pixelX = canvasCssPixelX(chart, native.clientX);
  }
  if (pixelX == null) return;
  const step = nearestStep(chart, pixelX);
  scheduleOpenNoteAtStep(step);
}

function attachFullscreenNoteHandlers(chart) {
  if (!chartIsAlive(chart)) return;
  applyNoteAnnotations(chart);
  const canvas = chart.canvas;
  // Re-bind cleanly across re-renders.
  if (noteCanvasClickBound && noteCanvasClickBound !== canvas) {
    noteCanvasClickBound.removeEventListener("pointerdown", onFullscreenCanvasPointerDown);
    noteCanvasClickBound.removeEventListener("click", onFullscreenCanvasClick);
  }
  canvas.removeEventListener("pointerdown", onFullscreenCanvasPointerDown);
  canvas.removeEventListener("click", onFullscreenCanvasClick);
  canvas.addEventListener("pointerdown", onFullscreenCanvasPointerDown);
  canvas.addEventListener("click", onFullscreenCanvasClick);
  noteCanvasClickBound = canvas;
  noteClickBoundCanvas = canvas;
  canvas.style.cursor = "crosshair";
}

function openNoteModal(step, { viewOnly = false } = {}) {
  pendingNoteStep = step;
  const modal = document.getElementById("noteModal");
  if (!modal) {
    console.error("noteModal element missing");
    return;
  }
  const stepEl = document.getElementById("noteModalStep");
  const listEl = document.getElementById("noteModalExisting");
  const form = document.getElementById("noteModalForm");
  const status = document.getElementById("noteModalStatus");

  if (stepEl) stepEl.textContent = `step ${step}`;
  if (status) {
    status.textContent = "";
    status.hidden = true;
  }

  const existing = notesForCurrentChart().filter((n) => n.step === step);
  if (listEl) {
    if (existing.length) {
      listEl.hidden = false;
      listEl.innerHTML =
        `<h4>Notes at this step</h4>` +
        existing
          .map(
            (n) => `
        <article class="note-card">
          <header>
            <span class="note-author">${escapeHtml(n.author)}</span>
            <time>${new Date(n.createdAt).toLocaleString()}</time>
          </header>
          <p>${escapeHtml(n.text)}</p>
        </article>`
          )
          .join("");
    } else {
      listEl.hidden = true;
      listEl.innerHTML = "";
    }
  }

  if (form) form.hidden = viewOnly;
  const authorEl = document.getElementById("noteAuthor");
  const textEl = document.getElementById("noteText");
  if (authorEl) authorEl.value = localStorage.getItem("noteAuthor") || "";
  if (textEl) textEl.value = "";

  modal.removeAttribute("hidden");
  modal.hidden = false;
  modal.classList.add("visible");
  modal.setAttribute("aria-hidden", "false");
  modal.style.display = "grid";
  modal.style.zIndex = "9999";
  if (!viewOnly) textEl?.focus();
}

function closeNoteModal() {
  const modal = document.getElementById("noteModal");
  if (!modal) return;
  modal.classList.remove("visible");
  modal.setAttribute("aria-hidden", "true");
  modal.hidden = true;
  modal.setAttribute("hidden", "");
  modal.style.display = "";
  pendingNoteStep = null;
}

async function submitNote(evt) {
  evt.preventDefault();
  const text = document.getElementById("noteText").value.trim();
  const author = document.getElementById("noteAuthor").value.trim() || "anonymous";
  const status = document.getElementById("noteModalStatus");
  const submitBtn = document.getElementById("noteSubmitBtn");
  if (!text) {
    status.hidden = false;
    status.textContent = "Please write a note before submitting.";
    return;
  }
  if (pendingNoteStep == null) return;

  localStorage.setItem("noteAuthor", author);
  const key = noteContextKey();
  submitBtn.disabled = true;
  status.hidden = false;
  status.textContent = "Publishing…";

  try {
    const result = await NotesStore.createNote({
      runId: key.runId,
      specId: key.specId,
      context: key.context,
      step: pendingNoteStep,
      author,
      text,
    });
    if (result.mode === "api") {
      allNotes.unshift(result.note);
      status.textContent = "Published.";
      if (chartIsAlive(fullChart)) applyNoteAnnotations(fullChart);
      setTimeout(closeNoteModal, 500);
    } else {
      status.textContent = "Continue in the new tab to publish, then refresh notes.";
    }
  } catch (err) {
    status.textContent = err.message || String(err);
  } finally {
    submitBtn.disabled = false;
  }
}

function wireNotesUi() {
  document.getElementById("noteModalClose")?.addEventListener("click", closeNoteModal);
  document.getElementById("noteModalCancel")?.addEventListener("click", closeNoteModal);
  document.getElementById("noteModal")?.addEventListener("click", (e) => {
    if (e.target.id === "noteModal") closeNoteModal();
  });
  document.getElementById("noteModalForm")?.addEventListener("submit", submitNote);
  document.getElementById("notesRefreshBtn")?.addEventListener("click", () => {
    refreshNotes();
  });
  document.getElementById("fullAddNoteBtn")?.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!fullOverlayOpen) return;
    openNoteModalAtChartCenter();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      const modal = document.getElementById("noteModal");
      if (modal && !modal.hidden) {
        e.stopImmediatePropagation();
        closeNoteModal();
      }
    }
  });
}
