# nanoGPT Observable Explorer

Interactive architecture viewer for nanoGPT training observables.

## Setup

After a training run produces files under `out/observables/`:

```bash
python3 viewer/scripts/build_viewer_data.py
```

This generates:

- `viewer/data/<run_id>/manifest.json` — module index + time series
- `viewer/data/<run_id>/curves/` — PNG curves (when available)
- `viewer/data/latest.json` — pointer to the newest run

## Local preview

Static files must be served over HTTP (browser blocks `fetch` on `file://`):

```bash
cd viewer
python3 -m http.server 8080
```

Open http://localhost:8080

## GitHub Pages

1. Push the `viewer/` folder to a GitHub repo (repo root can be the viewer folder contents).
2. Enable **Settings → Pages → Deploy from branch → main / root**.
3. After each training run, re-run `build_viewer_data.py` and commit updated `data/`.

## Usage

- Use **L0–L11** tabs to switch transformer blocks.
- Click any highlighted module to see observables grouped by weight / grad / update / activation.
- Click a chip to plot the curve (from CSV series now, PNG when training finishes).
- For block modules (`h.N.*`), enable **叠加各层对比** to overlay the same observable
  (same source / role / reduction / transforms) across all layers on one chart.



git push -u origin main --force
