#!/usr/bin/env python3
"""
Build viewer data from training observables: manifest + tree-organized curves.

Usage:
    python3 scripts/build_viewer_data.py
    python3 scripts/build_viewer_data.py --obs-dir ../out_baseline/observables
    python3 scripts/build_viewer_data.py --run run_20260709_031351
    python3 scripts/build_viewer_data.py --clean
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# Repo root is this viewer package (scripts/..). Also works when nested as
# <project>/viewer/scripts/... with observables under <project>/out_baseline.
VIEWER_ROOT = Path(__file__).resolve().parents[1]
VIEWER_DATA = VIEWER_ROOT / "data"
_PARENT = VIEWER_ROOT.parent
DEFAULT_OBS = (
    (_PARENT / "out_baseline" / "observables")
    if (_PARENT / "out_baseline" / "observables").is_dir()
    else (VIEWER_ROOT / "out_baseline" / "observables")
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curve_tree import organize_curves, selector_to_ui_module  # noqa: E402

_LAYER_RE = re.compile(r"^h\.(\d+)\.(.+)$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_run_id(run_id: str) -> str:
    if (
        not run_id
        or run_id in (".", "..")
        or not _RUN_ID_RE.fullmatch(run_id)
    ):
        raise ValueError(f"unsafe run id: {run_id!r}")
    return run_id


def parse_layer_role(ui_module: str) -> tuple[int | None, str]:
    m = _LAYER_RE.match(ui_module)
    if not m:
        return None, ui_module
    return int(m.group(1)), m.group(2)


def _temporal_label(temporal: list) -> str:
    parts = []
    for entry in temporal or []:
        if isinstance(entry, (list, tuple)) and len(entry) >= 1:
            name = str(entry[0])
            params = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
            args = ",".join(f"{key}={value}" for key, value in params.items())
            parts.append(f"{name}({args})")
        elif isinstance(entry, str):
            parts.append(entry)
    return ">".join(parts)


def temporal_from_canonical_id(canonical_id: str) -> list:
    parts = str(canonical_id or "").split("::", 4)
    if len(parts) < 5 or not parts[4] or parts[4] == "-":
        return []
    temporal = []
    for raw in parts[4].split("|"):
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", raw)
        if not match:
            temporal.append([raw, {}])
            continue
        params = {}
        if match.group(2):
            for pair in match.group(2).split(","):
                key, sep, value = pair.partition("=")
                if not sep:
                    continue
                try:
                    params[key.strip()] = json.loads(value)
                except json.JSONDecodeError:
                    params[key.strip()] = value.strip()
        temporal.append([match.group(1), params])
    return temporal


def family_id(
    source_kind: str,
    role: str,
    reduction: str,
    transforms: list,
    temporal: list,
) -> str:
    t = ">".join(transforms) if transforms else "-"
    base = f"{source_kind}|{role}|{reduction}|{t}"
    tp = _temporal_label(temporal)
    # Preserve historical family ids for the overwhelmingly common
    # non-temporal case; append the pipeline only when it distinguishes a
    # temporal family.
    return f"{base}|{tp}" if tp else base


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
        return [validate_run_id(explicit)]
    specs_files = sorted(obs_dir.glob("run_*_specs.json"), key=os.path.getmtime, reverse=True)
    if not specs_files:
        raise SystemExit(f"No specs found under {obs_dir}")
    return [p.name.replace("_specs.json", "") for p in specs_files]


def load_series(obs_dir: Path, run_id: str) -> dict[str, dict]:
    obs_path = obs_dir / f"{run_id}_observations.csv"
    values_by_step: dict[str, dict[int, float]] = defaultdict(dict)
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
            if not math.isfinite(val):
                continue
            if row.get("valid") not in (None, "", "ok"):
                continue
            # A resumed run can contain the same step more than once. The last
            # valid row is the newest value and should win deterministically.
            values_by_step[spec_id][step] = val
    return {
        spec_id: {
            "steps": sorted(points),
            "values": [points[step] for step in sorted(points)],
        }
        for spec_id, points in values_by_step.items()
    }


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
        cid = spec["canonical_id"]
        temporal = spec.get("temporal") or temporal_from_canonical_id(cid)
        transform_label = ">".join(transforms) if transforms else None
        temporal_label = _temporal_label(temporal) or None
        fid = family_id(
            spec["source_kind"], role, spec["reduction"], transforms, temporal
        )
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
            "temporal": temporal,
            "every": spec.get("every", 1),
            "label": _spec_label(spec, transform_label, temporal_label),
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
            "temporal": members[0]["temporal"],
            "label": members[0]["label"],
            "spec_ids": [m["id"] for m in members],
            "n_layers": len(members),
        }

    layer_idxs = [e["layer"] for e in spec_entries if e.get("layer") is not None]
    n_layer = (max(layer_idxs) + 1) if layer_idxs else 12

    return {
        "run_id": run_id,
        "model": {
            "name": "nanoGPT GPT-2 124M",
            "n_layer": n_layer,
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


def _spec_label(
    spec: dict,
    transform_label: str | None,
    temporal_label: str | None,
) -> str:
    parts = [spec["source_kind"], spec["reduction"]]
    if transform_label:
        parts.insert(1, transform_label)
    if temporal_label:
        parts.append(temporal_label)
    return " · ".join(parts)


def clean_viewer_data():
    if VIEWER_DATA.exists():
        shutil.rmtree(VIEWER_DATA)
    VIEWER_DATA.mkdir(parents=True)


def copy_loss_log(
    obs_dir: Path, run_id: str, dest: Path, *, allow_shared: bool = False
) -> bool:
    """Copy run-specific loss log into viewer data. Prefer per-run filenames."""
    candidates = [
        obs_dir / f"{run_id}_eval_loss_log.csv",
        obs_dir / run_id / "eval_loss_log.csv",
        dest / "eval_loss_log.csv",  # already present (manual drop-in)
    ]
    # Shared fallback only for single-run builds — otherwise every run
    # could silently receive the same loss curve.
    if allow_shared:
        candidates.append(obs_dir / "eval_loss_log.csv")
    for src in candidates:
        if not src.is_file():
            continue
        target = dest / "eval_loss_log.csv"
        if src.resolve() != target.resolve():
            shutil.copy2(src, target)
        return True
    return False


def build_run(obs_dir: Path, run_id: str, *, allow_shared_loss: bool = False) -> dict:
    run_id = validate_run_id(run_id)
    specs_path = obs_dir / f"{run_id}_specs.json"
    if not specs_path.exists():
        raise SystemExit(f"Missing {specs_path}")

    with specs_path.open(encoding="utf-8") as f:
        specs = json.load(f)["specs"]

    dest = VIEWER_DATA / run_id
    dest.mkdir(parents=True, exist_ok=True)
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

    has_loss = copy_loss_log(obs_dir, run_id, dest, allow_shared=allow_shared_loss)

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
    n_available = sum(
        1 for s in manifest["specs"] if s.get("series") or s.get("curve_png")
    )
    return {
        "run_id": run_id,
        "label": run_id,
        "n_specs": n_available,
        "n_curves": n_curves,
        "n_series": n_series,
        "provenance": manifest.get("provenance", {}),
    }


def load_existing_labels() -> dict[str, str]:
    """Preserve custom display names from data/index.json across rebuilds."""
    index_path = VIEWER_DATA / "index.json"
    if not index_path.exists():
        return {}
    try:
        with index_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    labels = {}
    for run in data.get("runs") or []:
        rid = run.get("run_id")
        lab = run.get("label")
        if rid and lab and lab != rid:
            labels[rid] = lab
    return labels


def rebuild_index(built_metas: list[dict]) -> list[dict]:
    """Merge newly built runs with any existing run dirs under viewer/data."""
    by_id = {m["run_id"]: m for m in built_metas}
    saved_labels = load_existing_labels()
    if VIEWER_DATA.exists():
        for run_dir in sorted(VIEWER_DATA.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
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
    for run_id, meta in by_id.items():
        if run_id in saved_labels:
            meta["label"] = saved_labels[run_id]
            continue
        # Folder may have been renamed; recover label via embedded manifest run_id.
        manifest_path = VIEWER_DATA / run_id / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            with manifest_path.open(encoding="utf-8") as f:
                old_id = json.load(f).get("run_id")
        except (OSError, json.JSONDecodeError):
            continue
        if old_id and old_id in saved_labels:
            meta["label"] = saved_labels[old_id]
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
    allow_shared_loss = len(run_ids) == 1
    metas = []
    for run_id in run_ids:
        meta = build_run(obs_dir, run_id, allow_shared_loss=allow_shared_loss)
        metas.append(meta)
        print(
            f"Built {run_id}: specs={meta['n_specs']}, "
            f"series={meta['n_series']}, tree curves={meta['n_curves']}"
        )

    index_metas = rebuild_index(metas)
    write_index(index_metas)
    print(f"Wrote {VIEWER_DATA / 'index.json'} ({len(index_metas)} run(s))")

    try:
        from build_reference_catalog import build_catalog

        build_catalog()
    except Exception as exc:  # noqa: BLE001 — catalog is optional for training rebuilds
        print(f"Warning: reference catalog not rebuilt ({exc})")


if __name__ == "__main__":
    main()
