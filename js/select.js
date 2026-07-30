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
      grid.appendChild(createRunCard(run, data.training_setup));
    }
  } catch (err) {
    loading.hidden = true;
    errorEl.hidden = false;
    errorEl.innerHTML =
      `Failed to load datasets: ${escapeHtml(err.message)}<br><br>` +
      `From the viewer root, run <code>python scripts/build_viewer_data.py --clean</code> first.`;
  }
}

function createRunCard(run, trainingSetup) {
  const card = document.createElement("article");
  card.className = "run-card";

  const link = document.createElement("a");
  link.className = "run-card-open";
  link.href =
    typeof CuratorUI !== "undefined"
      ? CuratorUI.withAdminParam(`explorer.html?run=${encodeURIComponent(run.run_id)}`)
      : `explorer.html?run=${encodeURIComponent(run.run_id)}`;

  const title = document.createElement("div");
  title.className = "run-card-title";
  title.textContent = run.label || run.run_id;

  const meta = document.createElement("div");
  meta.className = "run-card-meta";
  meta.innerHTML = `<span>${run.n_specs ?? "—"} observables</span>`;

  const action = document.createElement("span");
  action.className = "run-card-action";
  action.textContent = "Open explorer →";

  link.appendChild(title);
  link.appendChild(meta);
  link.appendChild(action);
  card.appendChild(link);

  if (trainingSetup && typeof TrainingConfig !== "undefined") {
    const details = document.createElement("details");
    details.className = "training-config training-config-card";
    const summary = document.createElement("summary");
    summary.textContent = "Inspect training config";
    const body = document.createElement("div");
    body.className = "training-config-body";
    TrainingConfig.render(body, trainingSetup, run);
    details.appendChild(summary);
    details.appendChild(body);
    card.appendChild(details);
  }

  return card;
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
