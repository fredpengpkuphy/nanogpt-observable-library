/**
 * Public site suggestions — anyone can post; only curator can reply (publicly).
 */
const SuggestionsUI = (() => {
  let items = [];
  let isAdmin = false;
  let replyOpenId = null;

  function escapeHtml(text) {
    return String(text)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function formatWhen(ts) {
    if (!ts) return "";
    try {
      const d = typeof ts.toDate === "function" ? ts.toDate() : new Date(ts);
      if (Number.isNaN(d.getTime())) return "";
      return d.toLocaleString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch (_) {
      return "";
    }
  }

  function setStatus(msg) {
    const el = document.getElementById("suggestStatus");
    if (el) el.textContent = msg || "";
  }

  function renderList() {
    const list = document.getElementById("suggestList");
    const empty = document.getElementById("suggestEmpty");
    if (!list) return;

    if (!items.length) {
      list.innerHTML = "";
      if (empty) empty.hidden = false;
      return;
    }
    if (empty) empty.hidden = true;

    list.innerHTML = items
      .map((s) => {
        const when = formatWhen(s.createdAt);
        const replies = (s.replies || [])
          .map(
            (r) => `
            <div class="suggest-reply">
              <div class="suggest-reply-meta">
                <span class="suggest-reply-label">Curator</span>
                ${whenReply(r)}
                ${
                  isAdmin
                    ? `<button type="button" class="chart-btn suggest-del-btn" data-del-reply="${escapeHtml(
                        s.id
                      )}" data-reply-id="${escapeHtml(r.id)}">Delete</button>`
                    : ""
                }
              </div>
              <p>${escapeHtml(r.text)}</p>
            </div>`
          )
          .join("");

        const adminReply =
          isAdmin && replyOpenId === s.id
            ? `<div class="suggest-reply-form">
                <textarea rows="2" maxlength="2000" data-reply-ta="${escapeHtml(
                  s.id
                )}" placeholder="Public curator reply…"></textarea>
                <div class="announce-item-actions">
                  <button type="button" class="chart-btn note-submit" data-send-reply="${escapeHtml(
                    s.id
                  )}">Reply</button>
                  <button type="button" class="chart-btn" data-cancel-reply>Cancel</button>
                </div>
              </div>`
            : isAdmin
              ? `<div class="announce-item-actions">
                  <button type="button" class="chart-btn" data-open-reply="${escapeHtml(
                    s.id
                  )}">Reply</button>
                  <button type="button" class="chart-btn announce-delete" data-del-suggest="${escapeHtml(
                    s.id
                  )}">Delete</button>
                </div>`
              : "";

        return `<article class="announce-item suggest-item" data-id="${escapeHtml(s.id)}">
          <div class="announce-item-meta">
            ${when ? `<time>${escapeHtml(when)}</time>` : `<span>Suggestion</span>`}
            <span class="announce-badge">public</span>
          </div>
          <p class="announce-item-text">${escapeHtml(s.text)}</p>
          ${replies ? `<div class="suggest-replies">${replies}</div>` : ""}
          ${adminReply}
        </article>`;
      })
      .join("");

    list.querySelectorAll("[data-open-reply]").forEach((btn) => {
      btn.addEventListener("click", () => {
        replyOpenId = btn.getAttribute("data-open-reply");
        renderList();
      });
    });
    list.querySelectorAll("[data-cancel-reply]").forEach((btn) => {
      btn.addEventListener("click", () => {
        replyOpenId = null;
        renderList();
      });
    });
    list.querySelectorAll("[data-send-reply]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const id = btn.getAttribute("data-send-reply");
        const ta = list.querySelector(`[data-reply-ta="${CSS.escape(id)}"]`);
        if (!ta) return;
        setStatus("Posting reply…");
        try {
          await NotesStore.replyToSuggestion(id, ta.value);
          replyOpenId = null;
          setStatus("Reply posted.");
          // Replies don't trigger parent snapshot; refresh once.
          items = await NotesStore.listSuggestions();
          renderList();
        } catch (err) {
          setStatus(err.message || String(err));
        }
      });
    });
    list.querySelectorAll("[data-del-suggest]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const id = btn.getAttribute("data-del-suggest");
        if (!confirm("Delete this suggestion and its replies?")) return;
        setStatus("Deleting…");
        try {
          await NotesStore.deleteSuggestion(id);
          setStatus("Deleted.");
        } catch (err) {
          setStatus(err.message || String(err));
        }
      });
    });
    list.querySelectorAll("[data-del-reply]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const sid = btn.getAttribute("data-del-reply");
        const rid = btn.getAttribute("data-reply-id");
        if (!confirm("Delete this reply?")) return;
        setStatus("Deleting…");
        try {
          await NotesStore.deleteSuggestionReply(sid, rid);
          items = await NotesStore.listSuggestions();
          setStatus("Reply deleted.");
          renderList();
        } catch (err) {
          setStatus(err.message || String(err));
        }
      });
    });
  }

  function whenReply(r) {
    const w = formatWhen(r.createdAt);
    return w ? `<time>${escapeHtml(w)}</time>` : "";
  }

  function wireComposer() {
    document.getElementById("suggestPublish")?.addEventListener("click", async () => {
      const ta = document.getElementById("suggestNewText");
      if (!ta) return;
      setStatus("Sending…");
      try {
        await NotesStore.createSuggestion(ta.value);
        ta.value = "";
        setStatus("Thanks — suggestion posted.");
      } catch (err) {
        setStatus(err.message || String(err));
      }
    });
  }

  function onAuth(e) {
    isAdmin = !!(e.detail && e.detail.isAdmin);
    renderList();
  }

  function wire() {
    if (typeof NotesStore === "undefined") return;
    document.addEventListener("curator-auth", onAuth);
    if (typeof CuratorUI !== "undefined" && CuratorUI.isAdmin) {
      isAdmin = CuratorUI.isAdmin();
    }
    wireComposer();
    NotesStore.watchSuggestions((list) => {
      items = list || [];
      renderList();
    });
    renderList();
  }

  return { wire };
})();
