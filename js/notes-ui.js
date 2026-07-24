/**
 * Fullscreen chart notes: general, single-step, or step-range tags plus comments/replies.
 * Anonymous posts (no public usernames). Curator can delete anything.
 */

let allNotes = [];
let notesReady = false;
let pendingNoteStep = null;
let noteClickTimer = null;
let noteClickBoundCanvas = null;
let noteIsAdmin = false;
let notePointerDown = null;
let noteCanvasClickBound = null;
let replyTarget = null; // { noteId, parentId }

function isNoteAdminEntry() {
  return typeof CuratorUI !== "undefined"
    ? CuratorUI.isAdminEntry()
    : new URLSearchParams(location.search).has("admin");
}

function clearNoteAdminEntry() {
  if (typeof CuratorUI !== "undefined") CuratorUI.clearAdminEntry();
}

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

function isUsableNoteStep(step) {
  if (typeof isValidStep === "function") return isValidStep(step);
  return Number.isInteger(step) && step >= 0 && step <= 1_000_000_000_000;
}

function isNoteModalOpen() {
  const modal = document.getElementById("noteModal");
  return !!(modal && !modal.hidden && modal.classList.contains("visible"));
}

async function refreshNotes() {
  try {
    allNotes = await NotesStore.listNotes(noteContextKey());
    notesReady = true;
  } catch (err) {
    console.warn(err);
    allNotes = [];
    notesReady = false;
  }
  updateNotesHint();
  if (fullOverlayOpen && chartIsAlive(fullChart)) {
    applyNoteAnnotations(fullChart);
  } else if (fullOverlayOpen) {
    renderNotesRail(notesForCurrentChart());
  }
}

async function deleteNoteById(id) {
  if (!id) return;
  try {
    await NotesStore.deleteNote(id);
    allNotes = allNotes.filter((n) => n.id !== id);
    if (chartIsAlive(fullChart)) applyNoteAnnotations(fullChart);
    else renderNotesRail(notesForCurrentChart());
  } catch (err) {
    alert(err.message || String(err));
  }
}

async function deleteCommentById(noteId, commentId) {
  if (!noteId || !commentId) return;
  try {
    await NotesStore.deleteComment(noteId, commentId);
    const note = allNotes.find((n) => n.id === noteId);
    if (note) {
      const drop = new Set([commentId]);
      let grew = true;
      while (grew) {
        grew = false;
        for (const c of note.comments || []) {
          if (c.parentId && drop.has(c.parentId) && !drop.has(c.id)) {
            drop.add(c.id);
            grew = true;
          }
        }
      }
      note.comments = (note.comments || []).filter((c) => !drop.has(c.id));
    }
    renderNotesRail(notesForCurrentChart());
  } catch (err) {
    alert(err.message || String(err));
  }
}

function updateNotesHint() {
  const hint = document.getElementById("fullChartHint");
  if (!hint || !fullOverlayOpen) return;
  const n = notesForCurrentChart().length;
  const base =
    "Drag a rectangle to zoom · click the curve to tag a step · notes can cover one step or a range · double-click to reset · Esc to close";
  hint.textContent = n ? `${base} · ${n} note${n === 1 ? "" : "s"}` : base;
}

function collectChartSteps(chart) {
  const steps = new Set();
  for (const ds of chart.data?.datasets || []) {
    if (ds._referenceLine) continue;
    for (const p of ds.data || []) {
      if (!p) continue;
      if (isUsableNoteStep(p._step)) steps.add(p._step);
      else if (isUsableNoteStep(p.x)) steps.add(p.x);
    }
  }
  return [...steps].sort((a, b) => a - b);
}

function nearestStep(chart, pixelX) {
  const xScale = chart.scales?.x;
  if (!xScale) return null;
  let best = null;
  let bestDist = Infinity;
  for (const ds of chart.data?.datasets || []) {
    if (ds._referenceLine) continue;
    for (const point of ds.data || []) {
      if (!point || !Number.isFinite(point.x)) continue;
      const step = isUsableNoteStep(point._step) ? point._step : point.x;
      if (!isUsableNoteStep(step)) continue;
      const pointPixel = xScale.getPixelForValue(point.x);
      if (!Number.isFinite(pointPixel)) continue;
      const distance = Math.abs(pointPixel - pixelX);
      if (distance < bestDist) {
        best = step;
        bestDist = distance;
      }
    }
  }
  return best;
}

function plotXForNoteStep(chart, step) {
  if (!isUsableNoteStep(step)) return NaN;
  for (const ds of chart?.data?.datasets || []) {
    if (ds._referenceLine) continue;
    const match = (ds.data || []).find((point) => point?._step === step);
    if (match && Number.isFinite(match.x)) return match.x;
  }
  const logarithmic = chart?.scales?.x?.type === "logarithmic";
  const axisMode =
    typeof fullOverlayMode !== "undefined" && fullOverlayMode === "loss"
      ? lossXAxisMode
      : curveXAxisMode;
  if (typeof xAxisValueForStep === "function" && axisMode === "tau") {
    const rid = typeof noteContextKey === "function" ? noteContextKey().runId : runId;
    const tau = xAxisValueForStep(step, rid, axisMode);
    if (Number.isFinite(tau)) return logarithmic && tau === 0 ? Number.EPSILON : tau;
  }
  return typeof plotXForStep === "function"
    ? plotXForStep(step, logarithmic)
    : logarithmic && step === 0
      ? 0.1
      : step;
}

function noteAnnotationConfig(notes, chart) {
  const byStep = new Map();
  const ranges = [];
  for (const n of notes) {
    if (!isUsableNoteStep(n.step)) continue;
    if (isUsableNoteStep(n.stepEnd) && n.stepEnd > n.step) {
      ranges.push(n);
      continue;
    }
    if (!byStep.has(n.step)) byStep.set(n.step, []);
    byStep.get(n.step).push(n);
  }
  const annotations = {};
  let i = 0;
  for (const note of ranges) {
    const xMin = plotXForNoteStep(chart, note.step);
    const xMax = plotXForNoteStep(chart, note.stepEnd);
    if (!Number.isFinite(xMin) || !Number.isFinite(xMax)) continue;
    annotations[`note_range_${i++}`] = {
      type: "box",
      xMin: Math.min(xMin, xMax),
      xMax: Math.max(xMin, xMax),
      backgroundColor: "rgba(232, 200, 136, 0.10)",
      borderColor: "rgba(232, 200, 136, 0.7)",
      borderWidth: 1,
      label: {
        display: true,
        content: `note · ${note.step}–${note.stepEnd}`,
        position: "start",
        backgroundColor: "rgba(14, 20, 25, 0.82)",
        color: "#ffe6b8",
        font: { size: 10, weight: "600" },
        padding: 3,
      },
    };
  }
  for (const [step, list] of byStep) {
    const x = plotXForNoteStep(chart, step);
    if (!Number.isFinite(x) || x < 0) continue;
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
    chart.options.plugins.annotation = {
      annotations: noteAnnotationConfig(notes, chart),
    };
    chart.update("none");
  } catch (err) {
    console.warn("applyNoteAnnotations failed", err);
  }
  updateNotesHint();
  renderNotesRail(notes);
}

function formatNoteTime(iso) {
  try {
    if (!iso) return "";
    const date = new Date(iso);
    return Number.isNaN(date.getTime()) ? "" : date.toLocaleString();
  } catch (_) {
    return "";
  }
}

function stepLabel(step, stepEnd = null) {
  if (!isUsableNoteStep(step)) return "General";
  return isUsableNoteStep(stepEnd) && stepEnd > step
    ? `steps ${step}–${stepEnd}`
    : `step ${step}`;
}

function buildCommentTree(comments) {
  const list = comments || [];
  const byParent = new Map();
  const knownIds = new Set(list.map((c) => c.id));
  const renderedIds = new Set();
  for (const c of list) {
    const key = c.parentId && knownIds.has(c.parentId) ? c.parentId : "";
    if (!byParent.has(key)) byParent.set(key, []);
    byParent.get(key).push(c);
  }
  function renderLevel(parentId, depth, ancestors = new Set()) {
    if (depth > 20) return "";
    const kids = byParent.get(parentId || "") || [];
    return kids
      .map((c) => {
        if (ancestors.has(c.id) || renderedIds.has(c.id)) return "";
        renderedIds.add(c.id);
        const nextAncestors = new Set(ancestors);
        nextAncestors.add(c.id);
        const nested = renderLevel(c.id, depth + 1, nextAncestors);
        return `
        <div class="note-comment ${depth ? "note-comment-reply" : ""}" data-comment-id="${escapeHtml(c.id)}">
          <div class="note-comment-head">
            <time>${escapeHtml(formatNoteTime(c.createdAt))}</time>
            <div class="note-comment-actions">
              <button type="button" class="note-reply-btn" data-note-id="${escapeHtml(c.noteId || "")}" data-parent-id="${escapeHtml(c.id)}">Reply</button>
              ${
                noteIsAdmin
                  ? `<button type="button" class="note-delete" data-note-id="${escapeHtml(c.noteId || "")}" data-comment-id="${escapeHtml(c.id)}" title="Delete comment">✕</button>`
                  : ""
              }
            </div>
          </div>
          <p>${escapeHtml(c.text)}</p>
          ${nested}
        </div>`;
      })
      .join("");
  }
  let html = renderLevel(null, 0);
  // Cycles have no natural root. Promote one member of each malformed
  // component so the remaining comments stay visible and recursion terminates.
  for (const comment of list) {
    if (renderedIds.has(comment.id)) continue;
    if (!byParent.has("")) byParent.set("", []);
    byParent.get("").push(comment);
    html += renderLevel(null, 0);
  }
  return html;
}

function renderNotesRail(notes) {
  const rail = document.getElementById("notesList");
  if (!rail) return;
  if (!notes.length) {
    rail.innerHTML = `<p class="notes-rail-empty">No notes yet. Write one above, or click the chart to tag a step.</p>`;
    return;
  }
  const sorted = [...notes].sort((a, b) => {
    const as = isUsableNoteStep(a.step) ? a.step : Number.POSITIVE_INFINITY;
    const bs = isUsableNoteStep(b.step) ? b.step : Number.POSITIVE_INFINITY;
    if (as !== bs) return as - bs;
    const ae = isUsableNoteStep(a.stepEnd) ? a.stepEnd : as;
    const be = isUsableNoteStep(b.stepEnd) ? b.stepEnd : bs;
    if (ae !== be) return ae - be;
    return String(a.createdAt).localeCompare(String(b.createdAt));
  });
  rail.innerHTML = sorted
    .map((n) => {
      const commentsHtml = buildCommentTree(
        (n.comments || []).map((c) => ({ ...c, noteId: n.id }))
      );
      return `
    <article class="note-card" data-id="${escapeHtml(n.id)}"
      data-step="${isUsableNoteStep(n.step) ? n.step : ""}"
      data-step-end="${isUsableNoteStep(n.stepEnd) ? n.stepEnd : ""}">
      <header>
        <span class="note-step">${escapeHtml(stepLabel(n.step, n.stepEnd))}</span>
        <time class="note-time">${escapeHtml(formatNoteTime(n.createdAt))}</time>
        ${
          noteIsAdmin
            ? `<button type="button" class="note-delete" data-id="${escapeHtml(n.id)}" title="Delete note">✕</button>`
            : ""
        }
      </header>
      <p>${escapeHtml(n.text)}</p>
      <div class="note-comments">${commentsHtml || ""}</div>
      <button type="button" class="note-reply-btn note-reply-root" data-note-id="${escapeHtml(n.id)}" data-parent-id="">Comment</button>
    </article>`;
    })
    .join("");

  rail.querySelectorAll(".note-delete[data-id]").forEach((btn) => {
    if (btn.dataset.commentId) return;
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (confirm("Delete this note and its comments?")) deleteNoteById(btn.dataset.id);
    });
  });
  rail.querySelectorAll(".note-delete[data-comment-id]").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (confirm("Delete this comment?")) {
        deleteCommentById(btn.dataset.noteId, btn.dataset.commentId);
      }
    });
  });
  rail.querySelectorAll(".note-reply-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      openReplyBox(btn.dataset.noteId, btn.dataset.parentId || null, btn);
    });
  });
}

function openReplyBox(noteId, parentId, anchorBtn) {
  document.querySelectorAll(".note-reply-box").forEach((el) => el.remove());
  replyTarget = { noteId, parentId: parentId || null };
  const box = document.createElement("form");
  box.className = "note-reply-box";
  box.innerHTML = `
    <textarea rows="2" maxlength="2000" placeholder="Write a reply…" required></textarea>
    <div class="note-reply-box-actions">
      <button type="button" class="chart-btn note-reply-cancel">Cancel</button>
      <button type="submit" class="chart-btn note-submit">Post</button>
    </div>
    <p class="note-reply-status" hidden></p>`;
  const host =
    anchorBtn.closest(".note-comment") || anchorBtn.closest(".note-card");
  if (host) host.appendChild(box);
  const ta = box.querySelector("textarea");
  ta?.focus();
  box.querySelector(".note-reply-cancel")?.addEventListener("click", () => {
    box.remove();
    replyTarget = null;
  });
  box.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = ta.value.trim();
    const status = box.querySelector(".note-reply-status");
    const submitBtn = box.querySelector("[type=submit]");
    if (!text) return;
    submitBtn.disabled = true;
    status.hidden = false;
    status.textContent = "Posting…";
    try {
      const comment = await NotesStore.createComment({
        noteId,
        text,
        parentId: parentId || null,
      });
      const note = allNotes.find((n) => n.id === noteId);
      if (note) {
        note.comments = note.comments || [];
        note.comments.push(comment);
      }
      replyTarget = null;
      renderNotesRail(notesForCurrentChart());
    } catch (err) {
      status.textContent = err.message || String(err);
      submitBtn.disabled = false;
    }
  });
}

function ceilNoteStepToData(value) {
  if (!Number.isFinite(value)) return null;
  const steps = chartIsAlive(fullChart) ? collectChartSteps(fullChart) : [];
  if (!steps.length) return Math.round(value);
  if (typeof ceilToAvailableStep === "function") {
    return ceilToAvailableStep(value, steps);
  }
  for (const s of steps) {
    if (s >= value) return s;
  }
  return steps[steps.length - 1];
}

function floorNoteStepToData(value) {
  if (!Number.isFinite(value)) return null;
  const steps = chartIsAlive(fullChart) ? collectChartSteps(fullChart) : [];
  if (!steps.length) return Math.round(value);
  for (let i = steps.length - 1; i >= 0; i -= 1) {
    if (steps[i] <= value) return steps[i];
  }
  return steps[0];
}

function railNoteMode() {
  const mode = document.getElementById("railNoteMode")?.value;
  return ["general", "single", "range"].includes(mode) ? mode : "general";
}

function updateRailNoteMode() {
  const mode = railNoteMode();
  const fields = document.getElementById("railNoteStepFields");
  const generalHint = document.getElementById("railNoteGeneralHint");
  const endWrap = document.getElementById("railNoteStepEndWrap");
  const startLabel = document.getElementById("railNoteStepLabel");
  const stepEl = document.getElementById("railNoteStep");
  const endEl = document.getElementById("railNoteStepEnd");
  if (fields) fields.hidden = mode === "general";
  if (generalHint) generalHint.hidden = mode !== "general";
  if (endWrap) endWrap.hidden = mode !== "range";
  if (startLabel) startLabel.textContent = mode === "range" ? "From" : "Step";
  if (stepEl) stepEl.required = mode !== "general";
  if (endEl) endEl.required = mode === "range";
}

function snapRailNoteStepInput(which = "start") {
  const stepEl = document.getElementById(
    which === "end" ? "railNoteStepEnd" : "railNoteStep"
  );
  if (!stepEl || stepEl.value.trim() === "") return null;
  const n = Number(stepEl.value);
  if (!Number.isFinite(n)) return null;
  const snapped =
    which === "end" ? floorNoteStepToData(n) : ceilNoteStepToData(n);
  if (snapped != null) stepEl.value = String(snapped);
  return snapped;
}

async function submitRailNote(evt) {
  evt.preventDefault();
  const textEl = document.getElementById("railNoteText");
  const stepEl = document.getElementById("railNoteStep");
  const stepEndEl = document.getElementById("railNoteStepEnd");
  const status = document.getElementById("railNoteStatus");
  const submitBtn = document.getElementById("railNoteSubmit");
  const text = textEl?.value.trim() || "";
  if (!text) {
    if (status) {
      status.hidden = false;
      status.textContent = "Please write a note.";
    }
    return;
  }
  const key = noteContextKey();
  if (!key.runId) {
    if (status) {
      status.hidden = false;
      status.textContent = "No run selected.";
    }
    return;
  }
  const mode = railNoteMode();
  let step = null;
  let stepEnd = null;
  if (mode !== "general") {
    if (!stepEl || stepEl.value.trim() === "") {
      status.hidden = false;
      status.textContent = mode === "range"
        ? "Choose the start step."
        : "Choose a step.";
      return;
    }
    const n = Number(stepEl.value);
    if (!Number.isFinite(n)) {
      status.hidden = false;
      status.textContent = "Step must be a number.";
      return;
    }
    step = ceilNoteStepToData(n);
    if (stepEl && step != null) stepEl.value = String(step);
  }
  if (mode === "range") {
    if (!stepEndEl || stepEndEl.value.trim() === "") {
      status.hidden = false;
      status.textContent = "Choose the end step.";
      return;
    }
    const n = Number(stepEndEl.value);
    if (!Number.isFinite(n)) {
      status.hidden = false;
      status.textContent = "End step must be a number.";
      return;
    }
    stepEnd = floorNoteStepToData(n);
    if (stepEndEl && stepEnd != null) stepEndEl.value = String(stepEnd);
    if (stepEnd <= step) {
      status.hidden = false;
      status.textContent = "The end step must be after the start step.";
      return;
    }
  }
  submitBtn.disabled = true;
  status.hidden = false;
  status.textContent = "Publishing…";
  try {
    const result = await NotesStore.createNote({
      runId: key.runId,
      specId: key.specId,
      context: key.context,
      step,
      stepEnd,
      text,
    });
    allNotes.unshift(result.note);
    textEl.value = "";
    if (stepEl) stepEl.value = "";
    if (stepEndEl) stepEndEl.value = "";
    const modeEl = document.getElementById("railNoteMode");
    if (modeEl) modeEl.value = "general";
    updateRailNoteMode();
    status.textContent = "Published.";
    if (chartIsAlive(fullChart)) applyNoteAnnotations(fullChart);
    else renderNotesRail(notesForCurrentChart());
    setTimeout(() => {
      status.hidden = true;
      status.textContent = "";
    }, 800);
  } catch (err) {
    status.textContent = err.message || String(err);
  } finally {
    submitBtn.disabled = false;
  }
}

function cancelPendingNoteClick() {
  if (noteClickTimer) {
    clearTimeout(noteClickTimer);
    noteClickTimer = null;
  }
}

function canvasCssPixelX(chart, clientX) {
  const canvas = chart?.canvas;
  if (!canvas) return null;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width) return null;
  return ((clientX - rect.left) / rect.width) * chart.width;
}

function scheduleOpenNoteAtStep(step) {
  if (!isUsableNoteStep(step)) return;
  cancelPendingNoteClick();
  noteClickTimer = setTimeout(() => {
    noteClickTimer = null;
    if (!fullOverlayOpen || !chartIsAlive(fullChart)) return;
    if (isNoteModalOpen()) return;
    // Prefer filling the rail form with this step.
    const modeEl = document.getElementById("railNoteMode");
    const stepEl = document.getElementById("railNoteStep");
    const textEl = document.getElementById("railNoteText");
    if (modeEl) modeEl.value = "single";
    updateRailNoteMode();
    if (stepEl) stepEl.value = String(Math.round(step));
    textEl?.focus();
  }, 280);
}

function onFullscreenCanvasPointerDown(evt) {
  if (evt.button != null && evt.button !== 0) return;
  notePointerDown = { x: evt.clientX, y: evt.clientY, t: Date.now() };
}

function onFullscreenCanvasClick(evt) {
  if (!fullOverlayOpen || !chartIsAlive(fullChart)) return;
  if (isNoteModalOpen()) return;
  if (evt.detail > 1) {
    cancelPendingNoteClick();
    return;
  }
  if (notePointerDown) {
    const dx = evt.clientX - notePointerDown.x;
    const dy = evt.clientY - notePointerDown.y;
    if (Math.hypot(dx, dy) >= 8) {
      cancelPendingNoteClick();
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

function openNoteModalAtChartCenter() {
  const textEl = document.getElementById("railNoteText");
  textEl?.focus();
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

function handleFullscreenChartClick(evt, chart) {
  if (!fullOverlayOpen || !chartIsAlive(chart)) return;
  if (isNoteModalOpen()) return;
  const native = evt?.native || evt;
  if (native?.detail > 1) {
    cancelPendingNoteClick();
    return;
  }
  if (notePointerDown && native?.clientX != null && native?.clientY != null) {
    const dx = native.clientX - notePointerDown.x;
    const dy = native.clientY - notePointerDown.y;
    if (Math.hypot(dx, dy) >= 8) {
      cancelPendingNoteClick();
      return;
    }
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

function openNoteModal(step) {
  // Legacy path: route into the rail composer.
  pendingNoteStep = step;
  const modeEl = document.getElementById("railNoteMode");
  const stepEl = document.getElementById("railNoteStep");
  const textEl = document.getElementById("railNoteText");
  if (modeEl && isUsableNoteStep(step)) modeEl.value = "single";
  updateRailNoteMode();
  if (stepEl && isUsableNoteStep(step)) stepEl.value = String(step);
  textEl?.focus();
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

let notesRailExpanded = false;

function isNotesRailExpanded() {
  return notesRailExpanded;
}

function setNotesRailExpanded(expanded) {
  notesRailExpanded = !!expanded;
  const modal = document.querySelector("#chartOverlay .chart-modal");
  const btn = document.getElementById("notesExpandBtn");
  if (modal) modal.classList.toggle("notes-expanded", notesRailExpanded);
  if (btn) {
    btn.textContent = notesRailExpanded ? "↘" : "⤢";
    btn.title = notesRailExpanded ? "Exit notes fullscreen" : "Expand notes";
  }
  // Chart was hidden; resize when returning so it doesn't stay blank.
  if (!notesRailExpanded && chartIsAlive(fullChart)) {
    try {
      fullChart.resize();
    } catch (_) {}
  }
}

function toggleNotesRailExpanded() {
  setNotesRailExpanded(!notesRailExpanded);
}

function renderAdminBar() {
  if (typeof CuratorUI !== "undefined") CuratorUI.render();
}

function onNotesAuthChange(state) {
  noteIsAdmin = !!state.isAdmin;
  renderAdminBar();
  const identityEl = document.getElementById("notesRailIdentity");
  if (identityEl) {
    identityEl.textContent = state.isAdmin ? "curator · can delete" : "";
  }
  if (fullOverlayOpen) {
    renderNotesRail(notesForCurrentChart());
  }
}

function wireNotesUi() {
  if (typeof CuratorUI !== "undefined") {
    document.addEventListener("curator-auth", (e) => {
      onNotesAuthChange({
        isAdmin: e.detail.isAdmin,
        handle: e.detail.handle,
      });
    });
    CuratorUI.wire();
  } else if (typeof NotesStore !== "undefined") {
    NotesStore.onAuthChange(onNotesAuthChange);
    NotesStore.init();
  }
  renderAdminBar();
  document.getElementById("railNoteForm")?.addEventListener("submit", submitRailNote);
  document.getElementById("railNoteMode")?.addEventListener("change", updateRailNoteMode);
  document.getElementById("railNoteStep")?.addEventListener(
    "change", () => snapRailNoteStepInput("start")
  );
  document.getElementById("railNoteStep")?.addEventListener(
    "blur", () => snapRailNoteStepInput("start")
  );
  document.getElementById("railNoteStepEnd")?.addEventListener(
    "change", () => snapRailNoteStepInput("end")
  );
  document.getElementById("railNoteStepEnd")?.addEventListener(
    "blur", () => snapRailNoteStepInput("end")
  );
  updateRailNoteMode();
  document.getElementById("noteModalClose")?.addEventListener("click", closeNoteModal);
  document.getElementById("noteModalCancel")?.addEventListener("click", closeNoteModal);
  document.getElementById("noteModal")?.addEventListener("click", (e) => {
    if (e.target.id === "noteModal") closeNoteModal();
  });
  document.getElementById("notesRefreshBtn")?.addEventListener("click", () => {
    refreshNotes();
  });
  document.getElementById("notesExpandBtn")?.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    toggleNotesRailExpanded();
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
        return;
      }
      if (notesRailExpanded) {
        e.stopImmediatePropagation();
        setNotesRailExpanded(false);
      }
    }
  });
}
