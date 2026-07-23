# nanoGPT Observable Explorer

Interactive architecture viewer for nanoGPT training observables.

## 1. Organize curves (optional standalone)

Flat PNGs under `out_baseline/observables/curve/` can be organized into a module tree:

```bash
python3 scripts/organize_curves.py --obs-dir ../out_baseline/observables
```

Output: `../out_baseline/observables/curve_tree/<run_id>/curves/...`

Tree layout:

```text
curves/{weight|grad|update|activation|attention|gelu_activation|...}/
  embeddings/{wte|wpe}/
  blocks/layer_{00..11}/{ln_1|attn|attn/c_attn|attn/c_proj|ln_2|mlp/c_fc|mlp/gelu|mlp/c_proj}/
  head/ln_f/
```

## 2. Build viewer data

Copies tree-organized curves + manifest into `data/`:

```bash
python3 scripts/build_viewer_data.py --clean
python3 scripts/build_viewer_data.py --obs-dir ../out_baseline/observables --run run_20260709_031351
```

Produces:

- `data/index.json` — list of all runs (dataset picker)
- `data/<run_id>/manifest.json` — specs, series, module index
- `data/<run_id>/curves/...` — tree-organized PNG curves
- `data/<run_id>/eval_loss_log.csv` — train/val loss (if available)

## 3. Local preview

```bash
python3 -m http.server 8080
```

- **http://localhost:8080/** — landing intro (`index.html`)
- **http://localhost:8080/select.html** — choose a training run
- **http://localhost:8080/explorer.html?run=…** — architecture explorer

## 4. GitHub Pages

Push this folder as the repo root. Entry point is the landing page `index.html`.  
After each training run, re-run `build_viewer_data.py` and commit `data/`.

## Usage

1. Open the site → **Start Exploration** → pick a run
2. Loss chart (if present): Both / Train / Val, step range filter, fullscreen
3. Use **L0–L11** tabs to switch transformer blocks
4. Click a module (incl. Attention entropy / GELU massive) → observables by source kind
5. Click a chip → interactive chart (CSV series) or PNG from the tree
6. Fullscreen a curve → **click any step** to leave a public note (stored in Firestore)

## Public notes (Firebase)

Notes, comments, suggestions, announcements, and maintenance state are stored in
Firebase Firestore. Visitors authenticate anonymously in the background; only the
configured curator account can edit or delete content.

See [`NOTES_SETUP.md`](NOTES_SETUP.md) for Firebase setup and publish the included
`firestore.rules` before enabling public posting.

## Tests

```bash
python -m unittest discover -s tests -v
```
