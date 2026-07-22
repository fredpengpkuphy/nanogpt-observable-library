async function bootSelect() {
  const loading = document.getElementById("selectLoading");
  const errorEl = document.getElementById("selectError");
  const grid = document.getElementById("runGrid");

  try {
    const res = await fetch(`data/index.json?t=${Date.now()}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`data/index.json (${res.status})`);
    const data = await res.json();
    const runs = data.runs || [];
    if (!runs.length) throw new Error("No runs in data/index.json");

    loading.hidden = true;
    grid.hidden = false;
    grid.innerHTML = "";

    for (const run of runs) {
      grid.appendChild(createRunCard(run));
    }
  } catch (err) {
    loading.hidden = true;
    errorEl.hidden = false;
    errorEl.innerHTML =
      `Failed to load datasets: ${escapeHtml(err.message)}<br><br>` +
      `From the viewer root, run <code>python scripts/build_viewer_data.py --clean</code> first.`;
  }
}

function createRunCard(run) {
  const btn = document.createElement("a");
  btn.className = "run-card";
  btn.href =
    typeof CuratorUI !== "undefined"
      ? CuratorUI.withAdminParam(`explorer.html?run=${encodeURIComponent(run.run_id)}`)
      : `explorer.html?run=${encodeURIComponent(run.run_id)}`;

  const title = document.createElement("div");
  title.className = "run-card-title";
  title.textContent = run.label || run.run_id;

  const meta = document.createElement("div");
  meta.className = "run-card-meta";
  meta.innerHTML = `<span>${run.n_specs ?? "—"} observables</span>`;

  btn.appendChild(title);
  btn.appendChild(meta);
  return btn;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

(async function startSelect() {
  if (typeof CuratorUI !== "undefined") CuratorUI.wire();
  if (typeof MaintenanceGate !== "undefined") {
    const ok = await MaintenanceGate.wire();
    if (!ok) return;
  }
  await bootSelect();
})();

