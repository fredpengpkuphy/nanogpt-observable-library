/**
 * Public site-wide announcement banner (Firestore meta/siteAnnouncement).
 */
const AnnouncementBanner = (() => {
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

  function render(text) {
    const el = ensureEl();
    const clean = String(text || "").trim();
    if (!clean) {
      el.hidden = true;
      el.innerHTML = "";
      document.body.classList.remove("has-announcement");
      return;
    }
    el.hidden = false;
    document.body.classList.add("has-announcement");
    el.innerHTML = `<span class="site-announcement-label">Announcement</span>`;
    const p = document.createElement("p");
    p.textContent = clean;
    el.appendChild(p);
  }

  function wire() {
    if (typeof NotesStore === "undefined") return;
    NotesStore.watchAnnouncement(render);
  }

  return { render, wire };
})();
