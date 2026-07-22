/**
 * Redirect non-curators away from gated pages when maintenance mode is on.
 * Include on select / explorer / reference (after NotesStore + CuratorUI).
 */
const MaintenanceGate = (() => {
  let decided = false;
  let watching = false;

  function maintenanceHref() {
    if (typeof CuratorUI !== "undefined" && CuratorUI.withAdminParam) {
      return CuratorUI.withAdminParam("maintenance.html");
    }
    try {
      if (sessionStorage.getItem("notesAdminEntry") === "1") {
        return "maintenance.html?admin=1";
      }
    } catch (_) {}
    return "maintenance.html";
  }

  function reveal() {
    document.documentElement.classList.remove("gate-pending");
  }

  function block() {
    decided = true;
    location.replace(maintenanceHref());
  }

  function startWatch() {
    if (watching || typeof NotesStore === "undefined") return;
    watching = true;
    NotesStore.watchMaintenanceMode((mode) => {
      if (mode.enabled && !NotesStore.isAdmin()) block();
    });
  }

  async function waitForAuth(timeoutMs = 2500) {
    if (typeof NotesStore === "undefined") return;
    await NotesStore.init();
    if (NotesStore.isAdmin()) return;
    await new Promise((resolve) => {
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        resolve();
      };
      const t = setTimeout(finish, timeoutMs);
      NotesStore.onAuthChange((state) => {
        if (state.ready || NotesStore.isReady() || state.isAdmin) {
          clearTimeout(t);
          finish();
        }
      });
    });
  }

  async function enforce() {
    if (decided) return !document.documentElement.classList.contains("gate-pending");
    if (typeof NotesStore === "undefined") {
      decided = true;
      reveal();
      return true;
    }
    try {
      await waitForAuth();
      const mode = await NotesStore.getMaintenanceMode();
      if (mode.enabled && !NotesStore.isAdmin()) {
        block();
        return false;
      }
    } catch (_) {
      /* fail open if backend unreachable */
    }
    decided = true;
    reveal();
    startWatch();
    return true;
  }

  function wire() {
    document.documentElement.classList.add("gate-pending");
    return enforce();
  }

  return { wire, enforce };
})();
