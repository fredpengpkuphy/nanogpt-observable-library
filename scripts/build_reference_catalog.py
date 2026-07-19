#!/usr/bin/env python3
"""Build a lightweight catalog of all observables for reference.html (no series)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VIEWER_ROOT = Path(__file__).resolve().parents[1]
VIEWER_DATA = VIEWER_ROOT / "data"


def pick_manifest() -> Path:
    index_path = VIEWER_DATA / "index.json"
    if index_path.exists():
        with index_path.open(encoding="utf-8") as f:
            runs = (json.load(f).get("runs") or [])
        for run in runs:
            rid = run.get("run_id")
            if rid:
                p = VIEWER_DATA / rid / "manifest.json"
                if p.exists():
                    return p
    for p in sorted(VIEWER_DATA.glob("*/manifest.json")):
        return p
    raise SystemExit(f"No manifest.json under {VIEWER_DATA}")


def build_catalog(
    manifest_path: Path | None = None,
    out_path: Path | None = None,
) -> int:
    manifest_path = manifest_path or pick_manifest()
    out_path = out_path or (VIEWER_DATA / "reference_catalog.json")
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    observables = []
    for s in manifest.get("specs") or []:
        observables.append(
            {
                "id": s["id"],
                "label": s.get("label") or s["id"],
                "selector": s.get("selector") or "",
                "source_kind": s.get("source_kind") or "",
                "reduction": s.get("reduction") or "",
                "transforms": s.get("transforms") or [],
                "ui_module": s.get("ui_module") or "",
                "layer": s.get("layer"),
                "role": s.get("role") or "",
                "every": s.get("every"),
            }
        )

    payload = {
        "n": len(observables),
        "source_manifest": str(manifest_path.relative_to(VIEWER_ROOT)).replace("\\", "/"),
        "run_id": manifest.get("run_id"),
        "observables": observables,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path} ({payload['n']} observables) from {manifest_path}")
    return payload["n"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=VIEWER_DATA / "reference_catalog.json",
    )
    args = parser.parse_args()
    build_catalog(args.manifest, args.out)


if __name__ == "__main__":
    main()
