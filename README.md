# nanoGPT Observable Explorer

Interactive architecture viewer for nanoGPT training observables.

## 1. Organize curves (optional standalone)

Flat PNGs under `out_baseline/observables/curve/` can be organized into a module tree:

```bash
python3 viewer/scripts/organize_curves.py --obs-dir out_baseline/observables
```

Output: `out_baseline/observables/curve_tree/<run_id>/curves/...`

Tree layout:

```text
curves/{weight|grad|update|activation}/
  embeddings/{wte|wpe}/
  blocks/layer_{00..11}/{ln_1|attn/c_attn|attn/c_proj|ln_2|mlp/c_fc|mlp/c_proj}/
  head/ln_f/
```

## 2. Build viewer data

Copies tree-organized curves + manifest into `viewer/data/`:

```bash
python3 viewer/scripts/build_viewer_data.py --clean
python3 viewer/scripts/build_viewer_data.py --obs-dir out_baseline/observables --run run_20260704_184808
```

Produces:

- `viewer/data/index.json` — list of all runs (dataset picker)
- `viewer/data/<run_id>/manifest.json` — specs, series, module index
- `viewer/data/<run_id>/curves/...` — tree-organized PNG curves

## 3. Local preview

```bash
cd viewer
python3 -m http.server 8080
```

- **http://localhost:8080/** — dataset selection (`index.html`)
- **http://localhost:8080/explorer.html?run=run_20260704_184808** — architecture explorer

## 4. GitHub Pages

Push the `viewer/` folder as repo root. Entry point is `index.html` (dataset picker).  
After each training run, re-run `build_viewer_data.py` and commit `data/`.

## Usage

1. Open the site → pick a run (e.g. `run_20260704_184808`)
2. Use **L0–L11** tabs to switch transformer blocks
3. Click a module → observables grouped by weight / grad / update / activation
4. Click a chip → interactive chart (CSV series) or PNG from the tree
