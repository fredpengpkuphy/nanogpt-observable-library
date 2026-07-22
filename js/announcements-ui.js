/**
 * Announcements page — public list + curator CRUD when signed in.
 */
const AnnouncementsUI = (() => {
  let items = [];
  let unsub = null;
  let isAdmin = false;
  let editingId = null;

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
    const el = document.getElementById("announcePageStatus");
    if (el) el.textContent = msg || "";
  }

  function renderList() {
    const list = document.getElementById("announceList");
    const empty = document.getElementById("announceEmpty");
    if (!list) return;

    if (!items.length) {
      list.innerHTML = "";
      if (empty) empty.hidden = false;
      return;
    }
    if (empty) empty.hidden = true;

    list.innerHTML = items
      .map((a) => {
        const when = formatWhen(a.updatedAt || a.createdAt);
        const legacy = a.legacy
          ? `<span class="announce-badge">legacy</span>`
          : "";
        const adminBtns = isAdmin
          ? `<div class="announce-item-actions">
              <button type="button" class="chart-btn" data-edit="${escapeHtml(a.id)}">Edit</button>
              <button type="button" class="chart-btn announce-delete" data-del="${escapeHtml(a.id)}">Delete</button>
            </div>`
          : "";
        const body =
          editingId === a.id && isAdmin
            ? `<textarea class="announce-edit-ta" data-edit-ta="${escapeHtml(
                a.id
              )}" rows="3" maxlength="1000">${escapeHtml(a.text)}</textarea>
               <div class="announce-item-actions">
                 <button type="button" class="chart-btn note-submit" data-save="${escapeHtml(
                   a.id
                 )}">Save</button>
                 <button type="button" class="chart-btn" data-cancel-edit>Cancel</button>
               </div>`
            : `<p class="announce-item-text">${escapeHtml(a.text)}</p>${adminBtns}`;

        return `<article class="announce-item" data-id="${escapeHtml(a.id)}">
          <div class="announce-item-meta">
            ${when ? `<time>${escapeHtml(when)}</time>` : `<span></span>`}
            ${legacy}
          </div>
          ${body}
        </article>`;
      })
      .join("");

    list.querySelectorAll("[data-edit]").forEach((btn) => {
      btn.addEventListener("click", () => {
        editingId = btn.getAttribute("data-edit");
        renderList();
      });
    });
    list.querySelectorAll("[data-cancel-edit]").forEach((btn) => {
      btn.addEventListener("click", () => {
        editingId = null;
        renderList();
      });
    });
    list.querySelectorAll("[data-save]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const id = btn.getAttribute("data-save");
        const ta = list.querySelector(`[data-edit-ta="${CSS.escape(id)}"]`);
        if (!ta) return;
        setStatus("Saving…");
        try {
          await NotesStore.updateAnnouncement(id, ta.value);
          editingId = null;
          setStatus("Saved.");
        } catch (err) {
          setStatus(err.message || String(err));
        }
      });
    });
    list.querySelectorAll("[data-del]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const id = btn.getAttribute("data-del");
        if (!confirm("Delete this announcement?")) return;
        setStatus("Deleting…");
        try {
          await NotesStore.deleteAnnouncement(id);
          if (editingId === id) editingId = null;
          setStatus("Deleted.");
        } catch (err) {
          setStatus(err.message || String(err));
        }
      });
    });
  }

  function renderComposer() {
    const panel = document.getElementById("announceComposer");
    if (!panel) return;
    panel.hidden = !isAdmin;
    const migrate = document.getElementById("announceMigrateWrap");
    if (migrate) {
      const hasLegacy = items.some((a) => a.legacy);
      migrate.hidden = !(isAdmin && hasLegacy);
    }
  }

  function render() {
    renderComposer();
    renderList();
  }

  function wireComposer() {
    document.getElementById("announcePublish")?.addEventListener("click", async () => {
      const ta = document.getElementById("announceNewText");
      if (!ta) return;
      setStatus("Publishing…");
      try {
        await NotesStore.createAnnouncement(ta.value);
        ta.value = "";
        setStatus("Published.");
      } catch (err) {
        setStatus(err.message || String(err));
      }
    });
    document.getElementById("announceMigrate")?.addEventListener("click", async () => {
      setStatus("Migrating…");
      try {
        await NotesStore.migrateLegacyAnnouncement();
        setStatus("Legacy announcement moved into the list.");
      } catch (err) {
        setStatus(err.message || String(err));
      }
    });
  }

  function onAuth(e) {
    isAdmin = !!(e.detail && e.detail.isAdmin);
    render();
  }

  function wire() {
    if (typeof NotesStore === "undefined") return;
    document.addEventListener("curator-auth", onAuth);
    if (typeof CuratorUI !== "undefined" && CuratorUI.isAdmin) {
      isAdmin = CuratorUI.isAdmin();
    }
    wireComposer();
    unsub = NotesStore.watchAnnouncements((list) => {
      items = list || [];
      render();
    });
    render();
  }

  return { wire };
})();
