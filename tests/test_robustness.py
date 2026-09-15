from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.multitask_gnn import CYPDMPNN
from models.transfer_model import load_encoder_weights
from src.constants import FEATURE_SCHEMA_VERSION, MODEL_CODE_VERSION
from src.data import DataBundle
from src.submission import SUBMISSION_COLUMNS, build_submission, validate_submission
from src.training import fit_target_scaling, train_dmpnn
from src.utils import RunContext, atomic_torch_save, seed_everything
from src.validation import audit_featurization, leakage_checks, validate_predictions


def bundle(frame: pd.DataFrame, train, validation) -> DataBundle:
    return DataBundle(
        frame=frame,
        train_mask=pd.Series(train, index=frame.index),
        validation_mask=pd.Series(validation, index=frame.index),
        task_names=("CYP1A2",), target_columns=("target",),
        lower_columns=("lower",), upper_columns=("upper",),
        uncertainty_columns=("sd",),
        split_metadata={"purpose": "direct_pic50", "outer_fold": 0},
    )


class RobustnessTests(unittest.TestCase):
    def test_evaluation_excludes_incomplete_run_from_metrics_and_oof(self):
        rows = pd.DataFrame(
            {
                "molecule_id": ["a", "b"], "fold": [0, 0],
                "CYP": ["CYP1A2", "CYP1A2"], "y_true": [1.0, 2.0],
                "y_pred": [1.1, 1.9], "lower": [0.9, 1.9], "upper": [1.1, 2.1],
                "seed": [1, 1], "model": ["model", "model"],
                "prediction_scale": ["original", "original"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, status in (("complete", "completed"), ("partial", "checkpointed")):
                run = root / name
                run.mkdir()
                rows.assign(y_pred=999.0 if name == "partial" else rows["y_pred"]).to_csv(
                    run / "predictions.csv", index=False
                )
                (run / "metadata.json").write_text(
                    json.dumps({"status": status, "experiment": "model", "fold": 0, "seed": 1})
                )
            output = root / "result"
            result = subprocess.run(
                [
                    sys.executable, "scripts/evaluate.py", "--input", str(root),
                    "--output-dir", str(output), "--aggregate-oof", "--folds", "0",
                ],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            oof = pd.read_csv(output / "oof_predictions.csv")
            self.assertEqual(len(oof), len(rows))
            self.assertLess(float(oof["y_pred"].max()), 10)

    def test_fold_leakage_detection_writes_summary(self):
        frame = pd.DataFrame(
            {
                "Molecule_Name": ["synthetic-a", "synthetic-b"],
                "canonical_smiles": ["CC", "CCC"], "target": [1.0, 2.0],
                "lower": [0.9, 1.9], "upper": [1.1, 2.1], "sd": [0.1, 0.1],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Leakage check failed"):
                leakage_checks(bundle(frame, [True, False], [True, True]), Path(directory))
            summary = json.loads((Path(directory) / "leakage_check.json").read_text())
            self.assertEqual(summary["status"], "failed")

    def test_target_normalization_and_inverse_transform(self):
        frame = pd.DataFrame(
            {
                "Molecule_Name": ["a", "b", "c"], "canonical_smiles": ["C", "CC", "CCC"],
                "target": [2.0, 4.0, 100.0], "lower": [1.0, 3.0, 99.0],
                "upper": [3.0, 5.0, 101.0], "sd": [1.0, 1.0, 1.0],
            }
        )
        scaling = fit_target_scaling(bundle(frame, [True, True, False], [False, False, True]))
        self.assertAlmostEqual(float(scaling.mean[0]), 3.0)
        normalized = (np.asarray([2.0, 4.0]) - scaling.mean[0]) / scaling.std[0]
        restored = normalized * scaling.std[0] + scaling.mean[0]
        np.testing.assert_allclose(restored, [2.0, 4.0])
        identity = fit_target_scaling(
            bundle(frame, [True, True, False], [False, False, True]), "none"
        )
        np.testing.assert_array_equal(identity.mean, [0.0])
        np.testing.assert_array_equal(identity.std, [1.0])

    def test_prediction_uniqueness_and_numeric_checks(self):
        valid = pd.DataFrame(
            {
                "molecule_id": ["a", "b"], "CYP": ["CYP1A2", "CYP1A2"],
                "y_pred": [1.0, 2.0], "prediction_scale": ["original", "original"],
            }
        )
        summary = validate_predictions(valid, {("a", "CYP1A2"), ("b", "CYP1A2")}, ["CYP1A2"], True)
        self.assertEqual(summary["rows"], 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_predictions(pd.concat([valid, valid.iloc[[0]]]), set(), ["CYP1A2"], True)

    def test_featurization_skip_is_logged_and_removed(self):
        frame = pd.DataFrame(
            {
                "Molecule_Name": ["valid", "invalid"],
                "canonical_smiles": ["CC", "not-a-smiles"], "SMILES": ["CC", "bad"],
                "target": [1.0, 2.0], "lower": [0.9, 1.9],
                "upper": [1.1, 2.1], "sd": [0.1, 0.1],
            }
        )
        data = bundle(frame, [True, True], [False, False])
        with tempfile.TemporaryDirectory() as directory:
            summary = audit_featurization(data, Path(directory), "skip_with_log")
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(int(data.train_mask.sum()), 1)
            failures = pd.read_csv(Path(directory) / "featurization_failures.csv")
            self.assertEqual(failures.loc[0, "molecule_id"], "invalid")

    def test_submission_schema_validation(self):
        test = pd.DataFrame({"SMILES": ["C", "CC"], "Molecule_Name": ["a", "b"]})
        predictions = pd.DataFrame(
            [
                {"molecule_id": name, "CYP": cyp, "y_pred": float(index + 1)}
                for index, name in enumerate(test["Molecule_Name"])
                for cyp in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
            ]
        )
        submission = build_submission(predictions, test)
        self.assertEqual(list(submission), SUBMISSION_COLUMNS)
        self.assertEqual(validate_submission(submission, test)["rows"], 2)

    def test_checkpoint_save_load_and_transfer_compatibility(self):
        model_config = {
            "type": "dmpnn", "message_hidden_dim": 16, "message_passing_depth": 3,
            "dropout": 0.1, "aggregation": "mean", "ffn_hidden_dim": 8,
            "ffn_num_layers": 2, "ffn_dropout": 0.1,
        }
        source = CYPDMPNN(("CYP1A2",), message_hidden_dim=16, ffn_hidden_dim=8)
        checkpoint = {
            "architecture_type": "dmpnn", "model_config": model_config,
            "task_names": ("CYP1A2",), "model_code_version": MODEL_CODE_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "pytorch_version": torch.__version__, "chemprop_version": "self-contained",
            "encoder_state_dict": source.encoder.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encoder.pt"
            atomic_torch_save(checkpoint, path)
            destination = CYPDMPNN(("CYP1A2",), message_hidden_dim=16, ffn_hidden_dim=8)
            metadata = load_encoder_weights(destination, path, model_config)
            self.assertGreater(metadata["loaded_key_count"], 0)
            incompatible = dict(model_config, message_hidden_dim=32)
            wrong_model = CYPDMPNN(("CYP1A2",), message_hidden_dim=32, ffn_hidden_dim=8)
            with self.assertRaisesRegex(ValueError, "Incompatible encoder checkpoint"):
                load_encoder_weights(wrong_model, path, incompatible)

    def test_resume_matches_uninterrupted_training(self):
        frame = pd.DataFrame(
            {
                "Molecule_Name": [f"m{i}" for i in range(12)],
                "canonical_smiles": ["C" * (i % 4 + 1) for i in range(12)],
                "target": np.linspace(1.0, 3.2, 12),
                "lower": np.linspace(0.9, 3.1, 12),
                "upper": np.linspace(1.1, 3.3, 12),
                "sd": np.repeat(0.1, 12),
            }
        )
        data = bundle(frame, [True] * 8 + [False] * 4, [False] * 8 + [True] * 4)
        config = {
            "experiment": "resume-test",
            "model": {
                "type": "dmpnn", "message_hidden_dim": 8,
                "message_passing_depth": 2, "dropout": 0.1, "aggregation": "mean",
                "ffn_hidden_dim": 8, "ffn_num_layers": 2, "ffn_dropout": 0.1,
            },
            "data": {"target_normalization": "zscore_per_target"},
            "training": {
                "batch_size": 4, "max_epochs": 2, "optimizer": "adamw",
                "learning_rate": 1e-3, "selection_metric": "macro_MAE",
                "metric_direction": "minimize", "precision": "fp32",
                "scheduler": "reduce_on_plateau", "scheduler_patience": 1,
            },
            "loss": {"loss_mode": "standard"},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed_everything(17)
            full_run = RunContext(root / "full", resume=False)
            train_dmpnn(data, config, full_run, 0, 17, "cpu", None)
            seed_everything(17)
            resumed_dir = root / "resumed"
            first = RunContext(resumed_dir, resume=False)
            first_result = train_dmpnn(data, config, first, 0, 17, "cpu", 0)
            self.assertTrue(first_result["runtime_limit_reached"])
            seed_everything(17)
            second = RunContext(resumed_dir, resume=True)
            train_dmpnn(data, config, second, 0, 17, "cpu", None)
            full = torch.load(full_run.checkpoints / "latest.pt", map_location="cpu", weights_only=False)
            resumed = torch.load(second.checkpoints / "latest.pt", map_location="cpu", weights_only=False)
            self.assertTrue(
                all(
                    torch.equal(full["model_state_dict"][key], resumed["model_state_dict"][key])
                    for key in full["model_state_dict"]
                )
            )


if __name__ == "__main__":
    unittest.main()
