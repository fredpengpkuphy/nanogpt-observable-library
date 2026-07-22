/**
 * Public site-wide announcement banner (Firestore announcements collection).
 * Shows the newest few items and links to announcements.html for the full list.
 */
const AnnouncementBanner = (() => {
  const BANNER_LIMIT = 3;

  function ensureEl() {
    let el = document.getElementById("siteAnnouncement");
    if (el) return el;
    el = document.createElement("aside");
    el.id = "siteAnnouncement";
    el.className = "site-announcement";
    el.setAttribute("role", "status");
    el.hidden = true;
    document.body.prepend(el);
    return el;
  }

  function escapeHtml(text) {
    return String(text)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function announcementsHref() {
    if (typeof CuratorUI !== "undefined" && CuratorUI.withAdminParam) {
      return CuratorUI.withAdminParam("announcements.html");
    }
    try {
      if (sessionStorage.getItem("notesAdminEntry") === "1") {
        return "announcements.html?admin=1";
      }
    } catch (_) {}
    return "announcements.html";
  }

  /** Accepts an array of {text} or a legacy single string. */
  function render(itemsOrText) {
    const el = ensureEl();
    let items = [];
    if (Array.isArray(itemsOrText)) {
      items = itemsOrText.filter((a) => a && String(a.text || "").trim());
    } else if (typeof itemsOrText === "string" && itemsOrText.trim()) {
      items = [{ text: itemsOrText.trim() }];
    }

    if (!items.length) {
      el.hidden = true;
      el.innerHTML = "";
      document.body.classList.remove("has-announcement");
      return;
    }

    el.hidden = false;
    document.body.classList.add("has-announcement");
    const shown = items.slice(0, BANNER_LIMIT);
    const more = items.length - shown.length;
    const href = announcementsHref();

    el.innerHTML = `
      <div class="site-announcement-head">
        <span class="site-announcement-label">Announcements</span>
        <a class="site-announcement-all" href="${href}">All${
          items.length > 1 ? ` (${items.length})` : ""
        } →</a>
      </div>
      <ul class="site-announcement-list">
        ${shown
          .map(
            (a) =>
              `<li><p>${escapeHtml(String(a.text).trim())}</p></li>`
          )
          .join("")}
      </ul>
      ${
        more > 0
          ? `<p class="site-announcement-more"><a href="${href}">+${more} more</a></p>`
          : ""
      }`;
  }

  function wire() {
    if (typeof NotesStore === "undefined") return;
    if (typeof NotesStore.watchAnnouncements === "function") {
      NotesStore.watchAnnouncements(render);
    } else {
      NotesStore.watchAnnouncement(render);
    }
  }

  return { render, wire };
})();
