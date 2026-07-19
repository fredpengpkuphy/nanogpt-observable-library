/**
 * Chart notes backed by GitHub Issues (public, persistent).
 * Issue title: [note] <runId> · step <n>
 * Issue body starts with a JSON meta fence, then free-text note.
 */

const NotesStore = (() => {
  const META_RE = /```note-meta\s*([\s\S]*?)```/;

  function cfg() {
    return window.NOTES_CONFIG || { enabled: false, github: {} };
  }

  function gh() {
    return cfg().github || {};
  }

  function apiBase() {
    const { owner, repo } = gh();
    return `https://api.github.com/repos/${owner}/${repo}`;
  }

  function headers(write = false) {
    const h = {
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
    };
    if (write && gh().token) h.Authorization = `Bearer ${gh().token}`;
    return h;
  }

  function encodeMeta(meta) {
    return "```note-meta\n" + JSON.stringify(meta, null, 2) + "\n```\n\n";
  }

  function parseIssue(issue) {
    const body = issue.body || "";
    const m = body.match(META_RE);
    let meta = {};
    let text = body;
    if (m) {
      try {
        meta = JSON.parse(m[1]);
      } catch (_) {
        meta = {};
      }
      text = body.slice(m.index + m[0].length).trim();
    }
    return {
      id: String(issue.number),
      issueNumber: issue.number,
      issueUrl: issue.html_url,
      runId: meta.runId || "",
      specId: meta.specId || "",
      context: meta.context || "spec",
      step: Number(meta.step),
      author: meta.author || issue.user?.login || "anonymous",
      text,
      createdAt: issue.created_at,
    };
  }

  async function listNotes() {
    if (!cfg().enabled) return [];
    const { label } = gh();
    const issues = [];
    // GitHub caps per_page at 100; page through so older notes are not dropped.
    for (let page = 1; page <= 20; page += 1) {
      const url =
        `${apiBase()}/issues?state=open&labels=${encodeURIComponent(label)}` +
        `&per_page=100&page=${page}&sort=created&direction=desc`;
      const res = await fetch(url, { headers: headers(false) });
      if (!res.ok) {
        console.warn("NotesStore.listNotes failed", res.status);
        break;
      }
      const batch = await res.json();
      if (!Array.isArray(batch) || !batch.length) break;
      issues.push(...batch);
      if (batch.length < 100) break;
    }
    return issues
      .filter((i) => !i.pull_request)
      .map(parseIssue)
      .filter((n) => Number.isFinite(n.step));
  }

  function filterNotes(notes, { runId, specId, context }) {
    return notes.filter(
      (n) =>
        n.runId === runId &&
        n.specId === specId &&
        (n.context || "spec") === (context || "spec")
    );
  }

  function buildIssuePayload({ runId, specId, context, step, author, text }) {
    const meta = {
      runId,
      specId,
      context: context || "spec",
      step: Math.round(step),
      author: (author || "anonymous").slice(0, 64),
    };
    const title = `[note] ${runId} · step ${meta.step}`;
    const body = encodeMeta(meta) + String(text || "").trim();
    return { title, body, labels: [gh().label || "chart-note"] };
  }

  async function createNote(note) {
    if (!cfg().enabled) throw new Error("Notes are disabled");
    const payload = buildIssuePayload(note);
    if (!gh().token) {
      const q = new URLSearchParams({
        title: payload.title,
        body: payload.body,
        labels: payload.labels.join(","),
      });
      const url = `https://github.com/${gh().owner}/${gh().repo}/issues/new?${q}`;
      window.open(url, "_blank", "noopener");
      return { mode: "github-form", url };
    }
    const res = await fetch(`${apiBase()}/issues`, {
      method: "POST",
      headers: { ...headers(true), "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`GitHub API ${res.status}: ${err.slice(0, 200)}`);
    }
    const issue = await res.json();
    return { mode: "api", note: parseIssue(issue) };
  }

  function hasWriteToken() {
    return Boolean(gh().token);
  }

  return { listNotes, filterNotes, createNote, hasWriteToken, cfg };
})();
