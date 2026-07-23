#!/usr/bin/env python3
"""Build a lightweight catalog of all observables for reference.html (no series)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

VIEWER_ROOT = Path(__file__).resolve().parents[1]
VIEWER_DATA = VIEWER_ROOT / "data"


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


def temporal_label(temporal: list) -> str:
    parts = []
    for entry in temporal:
        if isinstance(entry, (list, tuple)) and entry:
            name = str(entry[0])
            params = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
            args = ",".join(f"{key}={value}" for key, value in params.items())
            parts.append(f"{name}({args})")
        elif isinstance(entry, str):
            parts.append(entry)
    return " → ".join(parts)


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
    manifest_path = (manifest_path or pick_manifest()).resolve()
    out_path = (out_path or (VIEWER_DATA / "reference_catalog.json")).resolve()
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    observables = []
    for s in manifest.get("specs") or []:
        temporal = s.get("temporal") or temporal_from_canonical_id(s.get("id", ""))
        label = s.get("label") or s["id"]
        suffix = temporal_label(temporal)
        if suffix and not all(part in label for part in suffix.split(" → ")):
            label = f"{label} · {suffix}"
        observables.append(
            {
                "id": s["id"],
                "label": label,
                "selector": s.get("selector") or "",
                "source_kind": s.get("source_kind") or "",
                "reduction": s.get("reduction") or "",
                "transforms": s.get("transforms") or [],
                "temporal": temporal,
                "ui_module": s.get("ui_module") or "",
                "layer": s.get("layer"),
                "role": s.get("role") or "",
                "every": s.get("every"),
            }
        )

    try:
        source_manifest = str(manifest_path.relative_to(VIEWER_ROOT)).replace("\\", "/")
    except ValueError:
        source_manifest = str(manifest_path)

    payload = {
        "n": len(observables),
        "source_manifest": source_manifest,
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
