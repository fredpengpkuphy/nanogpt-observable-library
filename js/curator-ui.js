/**
 * Site-wide curator (admin) chrome — works on any page with #curatorBar.
 * Enter via ?admin=1 on the homepage, select page, or explorer.
 * Curators can also publish a site-wide announcement broadcast.
 */
const CuratorUI = (() => {
  let noteIsAdmin = false;
  let cachedAnnouncement = "";

  function isAdminEntry() {
    try {
      const params = new URLSearchParams(location.search);
      if (params.has("admin")) {
        sessionStorage.setItem("notesAdminEntry", "1");
        return true;
      }
      if (sessionStorage.getItem("notesAdminEntry") === "1") return true;
    } catch (_) {}
    return false;
  }

  function clearAdminEntry() {
    try {
      sessionStorage.removeItem("notesAdminEntry");
    } catch (_) {}
  }

  function showChrome() {
    return isAdminEntry() || noteIsAdmin;
  }

  /** Append ?admin=1 / &admin=1 when curator entry is active. */
  function withAdminParam(url) {
    if (!isAdminEntry() && !noteIsAdmin) return url;
    if (/[?&]admin=/.test(url)) return url;
    return url.includes("?") ? `${url}&admin=1` : `${url}?admin=1`;
  }

  function ensureBar() {
    let bar = document.getElementById("curatorBar");
    if (bar) return bar;
    bar = document.createElement("div");
    bar.className = "curator-bar";
    bar.id = "curatorBar";
    bar.hidden = true;
    bar.innerHTML = `<div class="curator-bar-inner" id="curatorBarInner"></div>`;
    const announce = document.getElementById("siteAnnouncement");
    if (announce && announce.parentNode === document.body) {
      announce.after(bar);
    } else {
      document.body.prepend(bar);
    }
    return bar;
  }

  async function refreshAnnouncementCache() {
    if (typeof NotesStore === "undefined") return;
    try {
      cachedAnnouncement = await NotesStore.getAnnouncement();
    } catch (_) {
      cachedAnnouncement = "";
    }
  }

  function wireAnnounceForm(inner) {
    const ta = inner.querySelector("#announceText");
    const status = inner.querySelector("#announceStatus");
    inner.querySelector("#announcePublish")?.addEventListener("click", async () => {
      status.textContent = "Publishing…";
      try {
        cachedAnnouncement = await NotesStore.setAnnouncement(ta.value);
        status.textContent = cachedAnnouncement ? "Published." : "Cleared.";
        if (typeof AnnouncementBanner !== "undefined") {
          AnnouncementBanner.render(cachedAnnouncement);
        }
      } catch (err) {
        status.textContent = err.message || String(err);
      }
    });
    inner.querySelector("#announceClear")?.addEventListener("click", async () => {
      status.textContent = "Clearing…";
      try {
        cachedAnnouncement = await NotesStore.setAnnouncement("");
        if (ta) ta.value = "";
        status.textContent = "Cleared.";
        if (typeof AnnouncementBanner !== "undefined") {
          AnnouncementBanner.render("");
        }
      } catch (err) {
        status.textContent = err.message || String(err);
      }
    });
  }

  function render() {
    ensureBar();
    const bar = document.getElementById("curatorBar");
    const inner = document.getElementById("curatorBarInner");
    if (!bar || !inner) return;

    if (!showChrome()) {
      bar.hidden = true;
      inner.innerHTML = "";
      document.body.classList.remove("curator-mode");
      return;
    }

    bar.hidden = false;
    if (noteIsAdmin) {
      document.body.classList.add("curator-mode");
      inner.innerHTML = `
        <div class="curator-bar-main">
          <div class="curator-bar-status">
            <span class="curator-pill">Curator mode</span>
            <span class="curator-hint">Delete notes · broadcast announcements</span>
          </div>
          <button type="button" class="chart-btn" id="notesAdminSignOut">Sign out</button>
        </div>
        <div class="curator-announce">
          <label for="announceText">Site announcement</label>
          <textarea id="announceText" rows="2" maxlength="1000" placeholder="Visible on every page…">${escapeHtml(
            cachedAnnouncement
          )}</textarea>
          <div class="curator-announce-actions">
            <button type="button" class="chart-btn note-submit" id="announcePublish">Publish</button>
            <button type="button" class="chart-btn" id="announceClear">Clear</button>
            <span class="notes-admin-status" id="announceStatus"></span>
          </div>
        </div>`;
      inner.querySelector("#notesAdminSignOut")?.addEventListener("click", async () => {
        clearAdminEntry();
        await NotesStore.signOutAdmin();
      });
      wireAnnounceForm(inner);
    } else {
      document.body.classList.remove("curator-mode");
      inner.innerHTML = `
        <div class="curator-bar-status">
          <span class="curator-pill curator-pill-muted">Curator sign-in</span>
          <span class="curator-hint">One login for the whole site</span>
        </div>
        <form class="notes-admin-form curator-form" id="notesAdminForm">
          <input type="email" id="adminEmail" placeholder="email" autocomplete="username" required />
          <input type="password" id="adminPass" placeholder="password" autocomplete="current-password" required />
          <button type="submit" class="chart-btn">Sign in</button>
          <span class="notes-admin-status" id="adminStatus"></span>
        </form>`;
      inner.querySelector("#notesAdminForm")?.addEventListener("submit", async (e) => {
        e.preventDefault();
        const email = inner.querySelector("#adminEmail").value.trim();
        const pass = inner.querySelector("#adminPass").value;
        const status = inner.querySelector("#adminStatus");
        status.textContent = "Signing in…";
        try {
          await NotesStore.signInAdmin(email, pass);
          status.textContent = "";
        } catch (err) {
          status.textContent = err.message || String(err);
        }
      });
    }
  }

  function escapeHtml(text) {
    return String(text)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function onAuthChange(state) {
    noteIsAdmin = !!state.isAdmin;
    if (noteIsAdmin) {
      try {
        sessionStorage.setItem("notesAdminEntry", "1");
      } catch (_) {}
      refreshAnnouncementCache().then(() => render());
    } else {
      render();
    }
    document.dispatchEvent(
      new CustomEvent("curator-auth", { detail: { isAdmin: noteIsAdmin, handle: state.handle } })
    );
  }

  function wire() {
    if (typeof NotesStore === "undefined") return;
    NotesStore.onAuthChange(onAuthChange);
    NotesStore.init();
    refreshAnnouncementCache().then(() => render());
    if (typeof AnnouncementBanner !== "undefined") AnnouncementBanner.wire();
  }

  function isAdmin() {
    return noteIsAdmin;
  }

  return { isAdminEntry, clearAdminEntry, withAdminParam, render, wire, isAdmin };
})();
