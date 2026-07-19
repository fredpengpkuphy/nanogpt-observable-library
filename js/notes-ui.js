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
    "单击某 step 留言 · 滚轮缩放 · 拖拽平移 · 双击重置视图 · Esc 关闭";
  hint.textContent = n
    ? `${base} · 本曲线 ${n} 条公开留言`
    : base;
}

function collectChartSteps(chart) {
  const steps = new Set();
  for (const ds of chart.data?.datasets || []) {
    for (const p of ds.data || []) {
      if (p && Number.isFinite(p.x)) steps.add(p.x);
    }
  }
  return [...steps].sort((a, b) => a - b);
}

function nearestStep(chart, pixelX) {
  const xScale = chart.scales?.x;
  if (!xScale) return null;
  const xVal = xScale.getValueForPixel(pixelX);
  if (!Number.isFinite(xVal)) return null;
  const steps = collectChartSteps(chart);
  if (!steps.length) return Math.round(xVal);
  let best = steps[0];
  let bestDist = Math.abs(steps[0] - xVal);
  for (const s of steps) {
    const d = Math.abs(s - xVal);
    if (d < bestDist) {
      best = s;
      bestDist = d;
    }
  }
  return best;
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
    annotations[`note_${i++}`] = {
      type: "line",
      xMin: step,
      xMax: step,
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
    rail.innerHTML = `<p class="notes-rail-empty">No public notes on this curve yet. Click the chart to add one.</p>`;
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

function detachFullscreenNoteHandlers(chart) {
  cancelPendingNoteClick();
  const canvas = chart?.canvas || noteClickBoundCanvas;
  if (canvas) {
    canvas.onclick = null;
    canvas.style.cursor = "";
  }
  noteClickBoundCanvas = null;
}

function attachFullscreenNoteHandlers(chart) {
  if (!chartIsAlive(chart)) return;
  applyNoteAnnotations(chart);
  const canvas = chart.canvas;
  // Replacing onclick avoids stacking handlers across re-renders.
  noteClickBoundCanvas = canvas;
  canvas.style.cursor = "crosshair";
  canvas.onclick = (evt) => {
    if (!fullOverlayOpen || !chartIsAlive(chart)) return;
    if (typeof isNoteModalOpen === "function" && isNoteModalOpen()) return;
    // Double-click: detail becomes 2 — cancel pending note open; dblclick resets view.
    if (evt.detail > 1) {
      cancelPendingNoteClick();
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const x = evt.clientX - rect.left;
    const step = nearestStep(chart, x);
    if (step == null) return;
    cancelPendingNoteClick();
    // Wait so a real double-click does not open the note modal.
    noteClickTimer = setTimeout(() => {
      noteClickTimer = null;
      if (!fullOverlayOpen || !chartIsAlive(fullChart)) return;
      if (typeof isNoteModalOpen === "function" && isNoteModalOpen()) return;
      openNoteModal(step);
    }, 280);
  };
}

function openNoteModal(step, { viewOnly = false } = {}) {
  pendingNoteStep = step;
  const modal = document.getElementById("noteModal");
  const stepEl = document.getElementById("noteModalStep");
  const listEl = document.getElementById("noteModalExisting");
  const form = document.getElementById("noteModalForm");
  const status = document.getElementById("noteModalStatus");
  if (!modal) return;

  stepEl.textContent = `step ${step}`;
  status.textContent = "";
  status.hidden = true;

  const existing = notesForCurrentChart().filter((n) => n.step === step);
  if (existing.length) {
    listEl.hidden = false;
    listEl.innerHTML =
      `<h4>Public notes at this step</h4>` +
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

  form.hidden = viewOnly;
  document.getElementById("noteAuthor").value =
    localStorage.getItem("noteAuthor") || "";
  document.getElementById("noteText").value = "";

  const tokenHint = document.getElementById("noteTokenHint");
  if (tokenHint) {
    tokenHint.hidden = NotesStore.hasWriteToken();
  }

  modal.hidden = false;
  modal.classList.add("visible");
  modal.setAttribute("aria-hidden", "false");
  if (!viewOnly) document.getElementById("noteText")?.focus();
}

function closeNoteModal() {
  const modal = document.getElementById("noteModal");
  if (!modal) return;
  modal.classList.remove("visible");
  modal.setAttribute("aria-hidden", "true");
  modal.hidden = true;
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
  status.textContent = NotesStore.hasWriteToken()
    ? "Publishing note…"
    : "Opening GitHub to publish your note…";

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
      status.textContent = "Note published.";
      if (chartIsAlive(fullChart)) applyNoteAnnotations(fullChart);
      setTimeout(closeNoteModal, 500);
    } else {
      status.textContent =
        "Finish creating the GitHub Issue, then refresh notes (or reopen fullscreen).";
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
