/**
 * Public chart notes (stored as GitHub Issues with label `chart-note`).
 *
 * Setup (once):
 * 1. In the GitHub repo → Issues → Labels → create label `chart-note`
 * 2. Create a Fine-grained PAT:
 *    https://github.com/settings/personal-access-tokens
 *    Repository access: this repo only
 *    Permissions: Issues → Read and write
 * 3. Paste the token below.
 *
 * Without a token, notes are still readable (public repo).
 * Submitting a note will open a prefilled GitHub Issue form instead.
 */
window.NOTES_CONFIG = {
  enabled: true,
  github: {
    owner: "fredpengpkuphy",
    repo: "nanogpt-observable-library",
    label: "chart-note",
    // Paste fine-grained PAT here (Issues: Read and write on this repo only).
    token: "",
  },
};
