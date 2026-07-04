#!/usr/bin/env python3
"""
Build viewer manifest + curve series from training outputs in out/observables/.

Usage:
    python3 viewer/scripts/build_viewer_data.py
    python3 viewer/scripts/build_viewer_data.py --run run_20260704_184808
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_OBS = ROOT / "out" / "observables"
VIEWER_DATA = ROOT / "viewer" / "data"


def selector_to_ui_module(selector: str) -> str:
    s = selector
    if s.startswith("transformer."):
        s = s[len("transformer.") :]
    if s.endswith(".weight"):
        s = s[: -len(".weight")]
    return s


_LAYER_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def parse_layer_role(ui_module: str) -> tuple[int | None, str]:
    """Split h.N.<role> into (layer_index, role). Non-block modules: (None, ui_module)."""
    m = _LAYER_RE.match(ui_module)
    if not m:
        return None, ui_module
    return int(m.group(1)), m.group(2)


def family_id(source_kind: str, role: str, reduction: str, transforms: list) -> str:
    """Same nature across layers share one family_id (layer index stripped)."""
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
    m = re.match(r"h\.(\d+)\.mlp\.c_fc$", ui_module)
    if m:
        return f"Block {m.group(1)} · MLP Up-Projection (c_fc)"
    m = re.match(r"h\.(\d+)\.mlp\.c_proj$", ui_module)
    if m:
        return f"Block {m.group(1)} · MLP Down-Projection"
    return ui_module


def find_run_id(explicit: str | None) -> str:
    if explicit:
        return explicit
    specs_files = sorted(OUT_OBS.glob("run_*_specs.json"), key=os.path.getmtime, reverse=True)
    if not specs_files:
        raise SystemExit(f"No specs found under {OUT_OBS}")
    return specs_files[0].name.replace("_specs.json", "")


def load_series(run_id: str) -> dict[str, dict]:
    obs_path = OUT_OBS / f"{run_id}_observations.csv"
    series: dict[str, dict] = defaultdict(lambda: {"steps": [], "values": []})
    if not obs_path.exists():
        return {}
    with obs_path.open(newline="") as f:
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


def copy_curve_pngs(run_id: str, dest_curves: Path, specs: list[dict]) -> None:
    src_curve = OUT_OBS / "curve"
    dest_curves.mkdir(parents=True, exist_ok=True)
    if not src_curve.exists():
        return
    for spec in specs:
        fname = spec.get("curve_file")
        if not fname:
            continue
        src = src_curve / fname
        if src.exists():
            shutil.copy2(src, dest_curves / fname)


def build_manifest(run_id: str) -> dict:
    specs_path = OUT_OBS / f"{run_id}_specs.json"
    if not specs_path.exists():
        raise SystemExit(f"Missing {specs_path}")

    with specs_path.open() as f:
        raw = json.load(f)

    series_map = load_series(run_id)
    modules: dict[str, dict] = {}
    spec_entries = []

    families: dict[str, list[str]] = defaultdict(list)

    for spec in raw["specs"]:
        ui_module = selector_to_ui_module(spec["selector"])
        layer, role = parse_layer_role(ui_module)
        transforms = spec.get("transforms") or []
        transform_label = ">".join(transforms) if transforms else None
        fid = family_id(spec["source_kind"], role, spec["reduction"], transforms)
        entry = {
            "id": spec["canonical_id"],
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
            "curve_png": f"curves/{spec['curve_file']}" if spec.get("curve_file") else None,
            "series": series_map.get(spec["canonical_id"]),
        }
        spec_entries.append(entry)
        families[fid].append(entry["id"])
        mod = modules.setdefault(
            ui_module,
            {"id": ui_module, "label": module_label(ui_module), "spec_ids": []},
        )
        mod["spec_ids"].append(entry["id"])

    # families: same nature across layers (layer index stripped), ≥2 members
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

    manifest = {
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
    return manifest


def _spec_label(spec: dict, transform_label: str | None) -> str:
    parts = [spec["source_kind"], spec["reduction"]]
    if transform_label:
        parts.insert(1, transform_label)
    return " · ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="run id, default: latest")
    args = parser.parse_args()

    run_id = find_run_id(args.run)
    dest = VIEWER_DATA / run_id
    dest.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(run_id)
    with (dest / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    copy_curve_pngs(run_id, dest / "curves", manifest["specs"])

    # pointer for the static app
    with (VIEWER_DATA / "latest.json").open("w") as f:
        json.dump({"run_id": run_id}, f)

    n_png = len(list((dest / "curves").glob("*.png"))) if (dest / "curves").exists() else 0
    n_series = sum(1 for s in manifest["specs"] if s.get("series"))
    n_families = len(manifest.get("families") or {})
    print(f"Built viewer data for {run_id}")
    print(f"  manifest: {dest / 'manifest.json'}")
    print(f"  specs: {manifest['n_specs']}, with series: {n_series}, "
          f"cross-layer families: {n_families}, png curves: {n_png}")


if __name__ == "__main__":
    main()
