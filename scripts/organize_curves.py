#!/usr/bin/env python3
"""
Organize flat observable curve PNGs into a module tree under out_baseline (or any obs dir).

Usage:
    python3 viewer/scripts/organize_curves.py --obs-dir out_baseline/observables
    python3 viewer/scripts/organize_curves.py --obs-dir out_baseline/observables --in-place
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from curve_tree import load_specs_from_obs_dir, organize_curves  # noqa: E402

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def find_run_ids(obs_dir: Path) -> list[str]:
    return sorted(
        {p.name.replace("_specs.json", "") for p in obs_dir.glob("run_*_specs.json")},
        reverse=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Organize flat curve/ PNGs into a module tree")
    parser.add_argument(
        "--obs-dir",
        type=Path,
        default=ROOT / "out_baseline" / "observables",
        help="directory with run_*_specs.json and curve/",
    )
    parser.add_argument("--run", default=None, help="run id; default: all runs found")
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="output tree root; default: <obs-dir>/curve_tree/<run_id>/curves",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="move PNGs from curve/ into curve_tree/<run_id>/curves (destructive)",
    )
    args = parser.parse_args()

    obs_dir = args.obs_dir.resolve()
    if not obs_dir.is_dir():
        raise SystemExit(f"obs-dir not found: {obs_dir}")

    run_ids = [args.run] if args.run else find_run_ids(obs_dir)
    if not run_ids:
        raise SystemExit(f"No run_*_specs.json under {obs_dir}")
    if args.in_place and len(run_ids) > 1:
        raise SystemExit(
            "--in-place cannot process multiple runs from one flat curve directory; "
            "pass --run explicitly"
        )

    src_curve = obs_dir / "curve"
    if not src_curve.is_dir():
        raise SystemExit(f"Missing flat curve dir: {src_curve}")

    for run_id in run_ids:
        if run_id in (".", "..") or not _RUN_ID_RE.fullmatch(run_id):
            raise SystemExit(f"Unsafe run id: {run_id!r}")
        specs = load_specs_from_obs_dir(obs_dir, run_id)
        if args.dest:
            dest_root = args.dest / run_id / "curves"
        else:
            dest_root = obs_dir / "curve_tree" / run_id / "curves"

        mapping = organize_curves(
            specs,
            src_curve,
            dest_root,
            copy=not args.in_place,
        )
        index_path = dest_root.parent / "curve_index.json"
        with index_path.open("w", encoding="utf-8") as f:
            json.dump({"run_id": run_id, "n_curves": len(mapping), "paths": mapping}, f, indent=2)

        print(f"{run_id}: organized {len(mapping)} curves -> {dest_root}")
        print(f"  index: {index_path}")


if __name__ == "__main__":
    main()
