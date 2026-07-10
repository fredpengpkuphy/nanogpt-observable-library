#!/usr/bin/env python3
"""
Build viewer data from training observables: manifest + tree-organized curves.

Usage:
    python3 viewer/scripts/build_viewer_data.py
    python3 viewer/scripts/build_viewer_data.py --obs-dir out_baseline/observables
    python3 viewer/scripts/build_viewer_data.py --run run_20260704_184808
    python3 viewer/scripts/build_viewer_data.py --clean
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIEWER_DATA = ROOT / "viewer" / "data"
DEFAULT_OBS = ROOT / "out_baseline" / "observables"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curve_tree import organize_curves, selector_to_ui_module  # noqa: E402

_LAYER_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def parse_layer_role(ui_module: str) -> tuple[int | None, str]:
    m = _LAYER_RE.match(ui_module)
    if not m:
        return None, ui_module
    return int(m.group(1)), m.group(2)


def family_id(source_kind: str, role: str, reduction: str, transforms: list) -> str:
    t = ">".join(transforms) if transforms else "-"
    return f"{source_kind}|{role}|{reduction}|{t}"


def module_label(ui_module: str) -> str:
    if ui_module == "wte":
        return "Token Embedding (wte)"
    if ui_module == "wpe":
        return "Position Embedding (wpe)"
    if ui_module == "ln_f":
        return "Final LayerNorm (ln_f)"
    m = re.match(r"h\.(\d+)\.ln_1$", ui_module)
    if m:
        return f"Block {m.group(1)} · Pre-Attn LayerNorm"
    m = re.match(r"h\.(\d+)\.ln_2$", ui_module)
    if m:
        return f"Block {m.group(1)} · Pre-MLP LayerNorm"
    m = re.match(r"h\.(\d+)\.attn\.c_attn$", ui_module)
    if m:
        return f"Block {m.group(1)} · Attention QKV Projection"
    m = re.match(r"h\.(\d+)\.attn\.c_proj$", ui_module)
    if m:
        return f"Block {m.group(1)} · Attention Output Projection"
    m = re.match(r"h\.(\d+)\.attn$", ui_module)
    if m:
        return f"Block {m.group(1)} · Attention (entropy / sink)"
    m = re.match(r"h\.(\d+)\.mlp\.c_fc$", ui_module)
    if m:
        return f"Block {m.group(1)} · MLP Up-Projection (c_fc)"
    m = re.match(r"h\.(\d+)\.mlp\.gelu$", ui_module)
    if m:
        return f"Block {m.group(1)} · MLP GELU (massive activation)"
    m = re.match(r"h\.(\d+)\.mlp\.c_proj$", ui_module)
    if m:
        return f"Block {m.group(1)} · MLP Down-Projection"
    if ui_module == "lm_head":
        return "LM Head (logits)"
    return ui_module


def find_run_ids(obs_dir: Path, explicit: str | None) -> list[str]:
    if explicit:
        return [explicit]
    specs_files = sorted(obs_dir.glob("run_*_specs.json"), key=os.path.getmtime, reverse=True)
    if not specs_files:
        raise SystemExit(f"No specs found under {obs_dir}")
    return [p.name.replace("_specs.json", "") for p in specs_files]


def load_series(obs_dir: Path, run_id: str) -> dict[str, dict]:
    obs_path = obs_dir / f"{run_id}_observations.csv"
    series: dict[str, dict] = defaultdict(lambda: {"steps": [], "values": []})
    if not obs_path.exists():
        return {}
    with obs_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            spec_id = row["spec_id"]
            if spec_id.startswith("loss::"):
                continue
            try:
                step = int(row["step"])
                val = float(row["value"])
            except (TypeError, ValueError):
                continue
            if row.get("valid") not in (None, "", "ok"):
                continue
            bucket = series[spec_id]
            bucket["steps"].append(step)
            bucket["values"].append(val)
    return dict(series)


def build_manifest(obs_dir: Path, run_id: str, curve_paths: dict[str, str]) -> dict:
    specs_path = obs_dir / f"{run_id}_specs.json"
    with specs_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    series_map = load_series(obs_dir, run_id)
    modules: dict[str, dict] = {}
    spec_entries = []
    families: dict[str, list[str]] = defaultdict(list)

    for spec in raw["specs"]:
        ui_module = selector_to_ui_module(spec["selector"])
        layer, role = parse_layer_role(ui_module)
        transforms = spec.get("transforms") or []
        transform_label = ">".join(transforms) if transforms else None
        fid = family_id(spec["source_kind"], role, spec["reduction"], transforms)
        cid = spec["canonical_id"]
        entry = {
            "id": cid,
            "ui_module": ui_module,
            "layer": layer,
            "role": role,
            "family_id": fid,
            "source_kind": spec["source_kind"],
            "selector": spec["selector"],
            "reduction": spec["reduction"],
            "transforms": transforms,
            "every": spec.get("every", 1),
            "label": _spec_label(spec, transform_label),
            "curve_png": curve_paths.get(cid),
            "series": series_map.get(cid),
        }
        spec_entries.append(entry)
        families[fid].append(cid)
        mod = modules.setdefault(
            ui_module,
            {"id": ui_module, "label": module_label(ui_module), "spec_ids": []},
        )
        mod["spec_ids"].append(cid)

    family_index: dict[str, dict] = {}
    by_id = {e["id"]: e for e in spec_entries}
    for fid, ids in families.items():
        members = [by_id[sid] for sid in ids if by_id[sid].get("layer") is not None]
        if len(members) < 2:
            continue
        members.sort(key=lambda m: m["layer"])
        family_index[fid] = {
            "id": fid,
            "role": members[0]["role"],
            "source_kind": members[0]["source_kind"],
            "reduction": members[0]["reduction"],
            "transforms": members[0]["transforms"],
            "label": members[0]["label"],
            "spec_ids": [m["id"] for m in members],
            "n_layers": len(members),
        }

    return {
        "run_id": run_id,
        "model": {
            "name": "nanoGPT GPT-2 124M",
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "block_size": 1024,
        },
        "provenance": raw.get("provenance", {}),
        "n_specs": len(spec_entries),
        "modules": modules,
        "families": family_index,
        "specs": spec_entries,
    }


def _spec_label(spec: dict, transform_label: str | None) -> str:
    parts = [spec["source_kind"], spec["reduction"]]
    if transform_label:
        parts.insert(1, transform_label)
    return " · ".join(parts)


def clean_viewer_data():
    if VIEWER_DATA.exists():
        shutil.rmtree(VIEWER_DATA)
    VIEWER_DATA.mkdir(parents=True)


def copy_loss_log(obs_dir: Path, run_id: str, dest: Path) -> bool:
    candidates = [
        obs_dir / "eval_loss_log.csv",
        obs_dir / f"{run_id}_eval_loss_log.csv",
        obs_dir.parent / "eval_loss_log.csv",
        dest / "eval_loss_log.csv",
    ]
    for src in candidates:
        if src.is_file():
            shutil.copy2(src, dest / "eval_loss_log.csv")
            return True
    return False


def build_run(obs_dir: Path, run_id: str) -> dict:
    specs_path = obs_dir / f"{run_id}_specs.json"
    if not specs_path.exists():
        raise SystemExit(f"Missing {specs_path}")

    with specs_path.open(encoding="utf-8") as f:
        specs = json.load(f)["specs"]

    dest = VIEWER_DATA / run_id
    curves_root = dest / "curves"
    if curves_root.exists():
        shutil.rmtree(curves_root)

    src_curve = obs_dir / "curve"
    curve_paths = {}
    if src_curve.is_dir():
        curve_paths = organize_curves(specs, src_curve, curves_root, copy=True)

    manifest = build_manifest(obs_dir, run_id, curve_paths)
    with (dest / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    has_loss = copy_loss_log(obs_dir, run_id, dest)

    meta = meta_from_manifest(run_id, manifest)
    meta["n_curves"] = len(curve_paths)
    meta["has_loss"] = has_loss
    return meta


def write_index(run_metas: list[dict]):
    with (VIEWER_DATA / "index.json").open("w", encoding="utf-8") as f:
        json.dump({"runs": run_metas}, f, indent=2)


def meta_from_manifest(run_id: str, manifest: dict) -> dict:
    n_series = sum(1 for s in manifest["specs"] if s.get("series"))
    n_curves = sum(1 for s in manifest["specs"] if s.get("curve_png"))
    return {
        "run_id": run_id,
        "label": run_id,
        "n_specs": manifest["n_specs"],
        "n_curves": n_curves,
        "n_series": n_series,
        "provenance": manifest.get("provenance", {}),
    }


def rebuild_index(built_metas: list[dict]) -> list[dict]:
    """Merge newly built runs with any existing run dirs under viewer/data."""
    by_id = {m["run_id"]: m for m in built_metas}
    if VIEWER_DATA.exists():
        for run_dir in sorted(VIEWER_DATA.glob("run_*")):
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name
            if run_id in by_id:
                continue
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            with manifest_path.open(encoding="utf-8") as f:
                manifest = json.load(f)
            meta = meta_from_manifest(run_id, manifest)
            meta["has_loss"] = (run_dir / "eval_loss_log.csv").is_file()
            by_id[run_id] = meta
    return sorted(by_id.values(), key=lambda m: m["run_id"], reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs-dir", type=Path, default=DEFAULT_OBS)
    parser.add_argument("--run", default=None, help="single run id; default: all runs in obs-dir")
    parser.add_argument("--clean", action="store_true", help="wipe viewer/data before build")
    args = parser.parse_args()

    obs_dir = args.obs_dir.resolve()
    if not obs_dir.is_dir():
        raise SystemExit(f"obs-dir not found: {obs_dir}")

    if args.clean or not (VIEWER_DATA / "index.json").exists():
        clean_viewer_data()
    else:
        VIEWER_DATA.mkdir(parents=True, exist_ok=True)

    run_ids = find_run_ids(obs_dir, args.run)
    metas = []
    for run_id in run_ids:
        meta = build_run(obs_dir, run_id)
        metas.append(meta)
        print(
            f"Built {run_id}: specs={meta['n_specs']}, "
            f"series={meta['n_series']}, tree curves={meta['n_curves']}"
        )

    index_metas = rebuild_index(metas)
    write_index(index_metas)
    print(f"Wrote {VIEWER_DATA / 'index.json'} ({len(index_metas)} run(s))")


if __name__ == "__main__":
    main()
