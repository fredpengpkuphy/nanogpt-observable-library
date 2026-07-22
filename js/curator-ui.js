/**
 * Site-wide curator (admin) chrome — compact top bar on pages with #curatorBar.
 * Enter via ?admin=1. Announcements / Suggestions / maintenance on their controls.
 */
const CuratorUI = (() => {
  let noteIsAdmin = false;
  let signInExpanded = false;
  let maintenanceOn = false;
  let maintBusy = false;

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

  async function refreshMaintenance() {
    if (typeof NotesStore === "undefined") return;
    try {
      const mode = await NotesStore.getMaintenanceMode();
      maintenanceOn = !!mode.enabled;
    } catch (_) {
      maintenanceOn = false;
    }
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
      const announceHref = withAdminParam("announcements.html");
      const suggestHref = withAdminParam("suggestions.html");
      const maintLabel = maintenanceOn ? "Maintenance ON" : "Maintenance OFF";
      const maintClass = maintenanceOn
        ? "chart-btn curator-bar-btn curator-maint-on"
        : "chart-btn curator-bar-btn";
      inner.innerHTML = `
        <div class="curator-bar-main">
          <div class="curator-bar-status">
            <span class="curator-pill">Curator</span>
            <a class="curator-link" href="${announceHref}">Announcements</a>
            <a class="curator-link" href="${suggestHref}">Suggestions</a>
            <button type="button" class="${maintClass}" id="maintToggle" title="Block public access to the explorer">${maintLabel}</button>
          </div>
          <button type="button" class="chart-btn curator-bar-btn" id="notesAdminSignOut">Sign out</button>
        </div>`;
      inner.querySelector("#notesAdminSignOut")?.addEventListener("click", async () => {
        clearAdminEntry();
        signInExpanded = false;
        await NotesStore.signOutAdmin();
      });
      inner.querySelector("#maintToggle")?.addEventListener("click", async () => {
        if (maintBusy) return;
        maintBusy = true;
        const btn = inner.querySelector("#maintToggle");
        if (btn) btn.disabled = true;
        try {
          maintenanceOn = await NotesStore.setMaintenanceMode(!maintenanceOn);
          render();
        } catch (err) {
          alert(err.message || "Could not update maintenance mode.");
          if (btn) btn.disabled = false;
        } finally {
          maintBusy = false;
        }
      });
    } else if (signInExpanded) {
      document.body.classList.remove("curator-mode");
      inner.innerHTML = `
        <div class="curator-bar-main">
          <span class="curator-pill curator-pill-muted">Curator</span>
          <form class="notes-admin-form curator-form" id="notesAdminForm">
            <input type="email" id="adminEmail" placeholder="email" autocomplete="username" required />
            <input type="password" id="adminPass" placeholder="password" autocomplete="current-password" required />
            <button type="submit" class="chart-btn curator-bar-btn">Sign in</button>
            <button type="button" class="chart-btn curator-bar-btn" id="adminSignInCancel">Cancel</button>
            <span class="notes-admin-status" id="adminStatus"></span>
          </form>
        </div>`;
      inner.querySelector("#adminSignInCancel")?.addEventListener("click", () => {
        signInExpanded = false;
        render();
      });
      inner.querySelector("#notesAdminForm")?.addEventListener("submit", async (e) => {
        e.preventDefault();
        const email = inner.querySelector("#adminEmail").value.trim();
        const pass = inner.querySelector("#adminPass").value;
        const status = inner.querySelector("#adminStatus");
        status.textContent = "…";
        try {
          await NotesStore.signInAdmin(email, pass);
          status.textContent = "";
        } catch (err) {
          status.textContent = err.message || String(err);
        }
      });
      inner.querySelector("#adminEmail")?.focus();
    } else {
      document.body.classList.remove("curator-mode");
      inner.innerHTML = `
        <div class="curator-bar-main">
          <div class="curator-bar-status">
            <span class="curator-pill curator-pill-muted">Curator</span>
          </div>
          <button type="button" class="chart-btn curator-bar-btn" id="adminSignInOpen">Sign in</button>
        </div>`;
      inner.querySelector("#adminSignInOpen")?.addEventListener("click", () => {
        signInExpanded = true;
        render();
      });
    }
  }

  function onAuthChange(state) {
    noteIsAdmin = !!state.isAdmin;
    if (noteIsAdmin) {
      try {
        sessionStorage.setItem("notesAdminEntry", "1");
      } catch (_) {}
      signInExpanded = false;
      refreshMaintenance().then(() => render());
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
    render();
    if (typeof AnnouncementBanner !== "undefined") AnnouncementBanner.wire();
    NotesStore.watchMaintenanceMode((mode) => {
      const next = !!mode.enabled;
      if (next === maintenanceOn) return;
      maintenanceOn = next;
      if (noteIsAdmin) render();
    });
  }

  function isAdmin() {
    return noteIsAdmin;
  }

  return { isAdminEntry, clearAdminEntry, withAdminParam, render, wire, isAdmin };
})();
