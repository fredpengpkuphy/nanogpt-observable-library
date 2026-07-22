/**
 * Site-wide curator (admin) chrome — works on any page with #curatorBar.
 * Enter via ?admin=1 on the homepage, select page, or explorer.
 */
const CuratorUI = (() => {
  let noteIsAdmin = false;

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

  function render() {
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
        <div class="curator-bar-status">
          <span class="curator-pill">Curator mode</span>
          <span class="curator-hint">Signed in site-wide — open any run / curve to delete notes</span>
        </div>
        <button type="button" class="chart-btn" id="notesAdminSignOut">Sign out</button>`;
      inner.querySelector("#notesAdminSignOut")?.addEventListener("click", async () => {
        clearAdminEntry();
        await NotesStore.signOutAdmin();
      });
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

  function onAuthChange(state) {
    noteIsAdmin = !!state.isAdmin;
    if (noteIsAdmin) {
      try {
        sessionStorage.setItem("notesAdminEntry", "1");
      } catch (_) {}
    }
    render();
    document.dispatchEvent(
      new CustomEvent("curator-auth", { detail: { isAdmin: noteIsAdmin, handle: state.handle } })
    );
  }

  function wire() {
    if (typeof NotesStore === "undefined") return;
    NotesStore.onAuthChange(onAuthChange);
    NotesStore.init();
    render();
  }

  function isAdmin() {
    return noteIsAdmin;
  }

  return { isAdminEntry, clearAdminEntry, withAdminParam, render, wire, isAdmin };
})();
