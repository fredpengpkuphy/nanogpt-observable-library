import csv
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from build_viewer_data import load_series, validate_run_id  # noqa: E402
from curve_tree import curve_tree_relpath  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from observable_lib import (  # noqa: E402
    ObservableEngine,
    ObservableSpec,
    TypedTensor,
    _r_std,
    check_spec,
)


class CurveTreeTests(unittest.TestCase):
    def test_normal_curve_path(self):
        self.assertEqual(
            curve_tree_relpath(
                "weight",
                "transformer.h.2.mlp.c_fc.weight",
                "curve.png",
            ),
            "weight/blocks/layer_02/mlp/c_fc/curve.png",
        )

    def test_curve_path_rejects_traversal(self):
        with self.assertRaises(ValueError):
            curve_tree_relpath("weight", "transformer.wte.weight", "../curve.png")
        with self.assertRaises(ValueError):
            curve_tree_relpath("../weight", "transformer.wte.weight", "curve.png")

    def test_run_id_rejects_traversal(self):
        self.assertEqual(validate_run_id("run_20260723"), "run_20260723")
        with self.assertRaises(ValueError):
            validate_run_id("../outside")


class ViewerDataTests(unittest.TestCase):
    def test_series_are_sorted_and_duplicate_steps_use_last_valid_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            obs_dir = Path(tmp)
            path = obs_dir / "demo_observations.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["step", "spec_id", "value", "valid"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"step": 20, "spec_id": "a", "value": 2, "valid": "ok"},
                        {"step": 10, "spec_id": "a", "value": 1, "valid": "ok"},
                        {"step": 20, "spec_id": "a", "value": 3, "valid": "ok"},
                        {"step": 30, "spec_id": "a", "value": 9, "valid": "nan"},
                        {"step": 40, "spec_id": "a", "value": "nan", "valid": "ok"},
                        {"step": 50, "spec_id": "a", "value": "inf", "valid": "ok"},
                    ]
                )
            self.assertEqual(
                load_series(obs_dir, "demo")["a"],
                {"steps": [10, 20], "values": [1.0, 3.0]},
            )


class ObservableTests(unittest.TestCase):
    def test_singleton_std_is_zero(self):
        tensor = TypedTensor(torch.tensor([3.0]), ("feature",), "x", "test")
        self.assertEqual(_r_std(tensor), 0.0)

    def test_schedule_must_be_positive(self):
        spec = ObservableSpec("weight", "weight", "mean", every=0)
        self.assertIn("positive integer", check_spec(spec))

    def test_temporal_parameters_are_validated(self):
        spec = ObservableSpec(
            "weight",
            "weight",
            "mean",
            temporal=(("slope", {"window": 1}),),
        )
        self.assertIn("window", check_spec(spec))

    def test_update_baselines_are_independent_per_cadence(self):
        model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        with tempfile.TemporaryDirectory() as tmp:
            engine = ObservableEngine(model, out_dir=tmp, run_id="test")
            engine.add_spec(
                ObservableSpec("update", "weight", "l2_norm", every=1)
            )
            engine.add_spec(
                ObservableSpec("update", "weight", "mean", every=2)
            )
            engine.freeze()

            self.assertEqual(engine.observe(0), [])
            with torch.no_grad():
                model.weight.add_(1)
            first = engine.observe(1)
            with torch.no_grad():
                model.weight.add_(1)
            second = engine.observe(2)

            self.assertAlmostEqual(first[0]["value"], 2.0)
            values = {row["spec_id"]: row["value"] for row in second}
            mean_id = next(key for key in values if "::mean::" in key)
            l2_id = next(key for key in values if "::l2_norm::" in key)
            self.assertAlmostEqual(values[l2_id], 2.0)
            self.assertAlmostEqual(values[mean_id], 2.0)
            engine.registry.close()

    def test_engine_requires_freeze_and_safe_run_id(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                ObservableEngine(model, out_dir=tmp, run_id="../outside")

            engine = ObservableEngine(model, out_dir=tmp, run_id="safe")
            engine.add_spec(ObservableSpec("weight", "weight", "mean"))
            with self.assertRaises(RuntimeError):
                engine.observe(1)
            engine.freeze()
            with self.assertRaises(RuntimeError):
                engine.add_spec(ObservableSpec("weight", "weight", "std"))
            engine.registry.close()


class ModelValidationTests(unittest.TestCase):
    def setUp(self):
        self.model = GPT(
            GPTConfig(
                block_size=4,
                vocab_size=8,
                n_layer=1,
                n_head=1,
                n_embd=4,
            )
        )
        self.tokens = torch.zeros((1, 1), dtype=torch.long)

    def test_generate_rejects_invalid_sampling_parameters(self):
        with self.assertRaises(ValueError):
            self.model.generate(self.tokens, 1, temperature=0)
        with self.assertRaises(ValueError):
            self.model.generate(self.tokens, 1, temperature=float("nan"))
        with self.assertRaises(ValueError):
            self.model.generate(self.tokens, 1, top_k=0)
        with self.assertRaises(ValueError):
            self.model.generate(self.tokens, -1)

    def test_crop_rejects_non_positive_size(self):
        with self.assertRaises(ValueError):
            self.model.crop_block_size(0)

    def test_forward_rejects_invalid_shapes(self):
        with self.assertRaises(ValueError):
            self.model(torch.zeros((0,), dtype=torch.long))
        with self.assertRaises(ValueError):
            self.model(torch.zeros((1, 0), dtype=torch.long))
        with self.assertRaises(ValueError):
            self.model(
                torch.zeros((1, 2), dtype=torch.long),
                torch.zeros((1, 1), dtype=torch.long),
            )


if __name__ == "__main__":
    unittest.main()
