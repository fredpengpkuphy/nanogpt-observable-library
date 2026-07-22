/**
 * Fullscreen chart notes: panel compose, optional step tags, comments/replies.
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

function isNoteModalOpen() {
  const modal = document.getElementById("noteModal");
  return !!(modal && !modal.hidden && modal.classList.contains("visible"));
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
    "Write in the Notes panel · click the curve to tag a step · scroll to zoom · double-click to reset · Esc to close";
  hint.textContent = n ? `${base} · ${n} note${n === 1 ? "" : "s"}` : base;
}

function collectChartSteps(chart) {
  const steps = new Set();
  for (const ds of chart.data?.datasets || []) {
    for (const p of ds.data || []) {
      if (!p) continue;
      if (Number.isFinite(p._step)) steps.add(p._step);
      else if (Number.isFinite(p.x)) steps.add(p.x);
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
  // Prefer matching against true steps (_step); plot may put step 0 at x=1 on log.
  if (!steps.length) return Math.round(xVal);
  let best = steps[0];
  let bestDist = Infinity;
  for (const s of steps) {
    const plotX =
      xScale.type === "logarithmic" && s === 0 ? 1 : s;
    const d = Math.abs(plotX - xVal);
    if (d < bestDist) {
      best = s;
      bestDist = d;
    }
  }
  return best;
}

function plotXForNoteStep(chart, step) {
  if (!Number.isFinite(step)) return step;
  // Only step 0 is remapped for log-x; everything else stays as-is.
  if (chart?.scales?.x?.type === "logarithmic" && step === 0) return 1;
  return step;
}

function noteAnnotationConfig(notes, chart) {
  const byStep = new Map();
  for (const n of notes) {
    if (!Number.isFinite(n.step)) continue;
    if (!byStep.has(n.step)) byStep.set(n.step, []);
    byStep.get(n.step).push(n);
  }
  const annotations = {};
  let i = 0;
  for (const [step, list] of byStep) {
    const x = plotXForNoteStep(chart, step);
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
    return new Date(iso).toLocaleString();
  } catch (_) {
    return "";
  }
}

function stepLabel(step) {
  return Number.isFinite(step) ? `step ${step}` : "General";
}

function buildCommentTree(comments) {
  const list = comments || [];
  const byParent = new Map();
  for (const c of list) {
    const key = c.parentId || "";
    if (!byParent.has(key)) byParent.set(key, []);
    byParent.get(key).push(c);
  }
  function renderLevel(parentId, depth) {
    const kids = byParent.get(parentId || "") || [];
    return kids
      .map((c) => {
        const nested = renderLevel(c.id, depth + 1);
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
  return renderLevel(null, 0);
}

function renderNotesRail(notes) {
  const rail = document.getElementById("notesList");
  if (!rail) return;
  if (!notes.length) {
    rail.innerHTML = `<p class="notes-rail-empty">No notes yet. Write one above, or click the chart to tag a step.</p>`;
    return;
  }
  const sorted = [...notes].sort((a, b) => {
    const as = Number.isFinite(a.step) ? a.step : Number.POSITIVE_INFINITY;
    const bs = Number.isFinite(b.step) ? b.step : Number.POSITIVE_INFINITY;
    if (as !== bs) return as - bs;
    return String(a.createdAt).localeCompare(String(b.createdAt));
  });
  rail.innerHTML = sorted
    .map((n) => {
      const commentsHtml = buildCommentTree(
        (n.comments || []).map((c) => ({ ...c, noteId: n.id }))
      );
      return `
    <article class="note-card" data-id="${escapeHtml(n.id)}" data-step="${Number.isFinite(n.step) ? n.step : ""}">
      <header>
        <span class="note-step">${escapeHtml(stepLabel(n.step))}</span>
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

async function submitRailNote(evt) {
  evt.preventDefault();
  const textEl = document.getElementById("railNoteText");
  const stepEl = document.getElementById("railNoteStep");
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
  let step = null;
  if (stepEl && stepEl.value.trim() !== "") {
    const n = Number(stepEl.value);
    if (!Number.isFinite(n)) {
      status.hidden = false;
      status.textContent = "Step must be a number (or leave blank).";
      return;
    }
    step = n;
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
      text,
    });
    allNotes.unshift(result.note);
    textEl.value = "";
    if (stepEl) stepEl.value = "";
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
  if (step == null || !Number.isFinite(step)) return;
  cancelPendingNoteClick();
  noteClickTimer = setTimeout(() => {
    noteClickTimer = null;
    if (!fullOverlayOpen || !chartIsAlive(fullChart)) return;
    if (isNoteModalOpen()) return;
    // Prefer filling the rail form with this step.
    const stepEl = document.getElementById("railNoteStep");
    const textEl = document.getElementById("railNoteText");
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
  const stepEl = document.getElementById("railNoteStep");
  const textEl = document.getElementById("railNoteText");
  if (stepEl && Number.isFinite(step)) stepEl.value = String(Math.round(step));
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
  document.getElementById("noteModalClose")?.addEventListener("click", closeNoteModal);
  document.getElementById("noteModalCancel")?.addEventListener("click", closeNoteModal);
  document.getElementById("noteModal")?.addEventListener("click", (e) => {
    if (e.target.id === "noteModal") closeNoteModal();
  });
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
