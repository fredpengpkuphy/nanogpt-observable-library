"""
Map observable specs to a hierarchical curve directory tree.

Tree layout (mirrors nanoGPT module paths from model.py + observable_lib selectors):

  curves/{source_kind}/embeddings/{wte|wpe}/{filename}
  curves/{source_kind}/blocks/layer_{NN}/{role/path}/{filename}
  curves/{source_kind}/head/ln_f/{filename}
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

_LAYER_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def selector_to_ui_module(selector: str) -> str:
    s = selector
    if s.startswith("transformer."):
        s = s[len("transformer.") :]
    if s.endswith(".weight"):
        s = s[: -len(".weight")]
    return s


def curve_tree_relpath(source_kind: str, selector: str, curve_filename: str) -> str:
    """Relative path under ``curves/`` for one PNG."""
    ui = selector_to_ui_module(selector)
    fname = curve_filename

    if ui in ("wte", "wpe"):
        return f"{source_kind}/embeddings/{ui}/{fname}"

    if ui == "ln_f":
        return f"{source_kind}/head/ln_f/{fname}"

    m = _LAYER_RE.match(ui)
    if m:
        layer = int(m.group(1))
        role = m.group(2).replace(".", "/")
        return f"{source_kind}/blocks/layer_{layer:02d}/{role}/{fname}"

    safe = ui.replace(".", "_")
    return f"{source_kind}/other/{safe}/{fname}"


def curve_tree_relpath_from_spec(spec: dict) -> str:
    curve_file = spec.get("curve_file")
    if not curve_file:
        raise ValueError(f"spec missing curve_file: {spec.get('canonical_id')}")
    return curve_tree_relpath(spec["source_kind"], spec["selector"], curve_file)


def organize_curves(
    specs: list[dict],
    src_curve_dir: Path,
    dest_curve_root: Path,
    *,
    copy: bool = True,
) -> dict[str, str]:
    """
    Copy or move flat PNGs into ``dest_curve_root`` using the tree layout.

    Returns mapping canonical_id -> relative path from run dir (``curves/...``).
    """
    dest_curve_root.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}

    for spec in specs:
        fname = spec.get("curve_file")
        if not fname:
            continue
        src = src_curve_dir / fname
        if not src.exists():
            continue
        rel = curve_tree_relpath(spec["source_kind"], spec["selector"], fname)
        dest = dest_curve_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if copy:
            shutil.copy2(src, dest)
        else:
            shutil.move(str(src), str(dest))
        mapping[spec["canonical_id"]] = f"curves/{rel}"

    return mapping


def load_specs_from_obs_dir(obs_dir: Path, run_id: str) -> list[dict]:
    specs_path = obs_dir / f"{run_id}_specs.json"
    with specs_path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return raw["specs"]
