from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import pandas as pd

from src.model_registry import MODEL_STATISTICS_COLUMNS, update_model_statistics
from src.run_management import finish_run
from src.utils import RunContext


class ModelRegistryTests(unittest.TestCase):
    @staticmethod
    def write_run(root: Path, name: str, fold: int, seed: int, st_rae: float) -> Path:
        run = root / name / f"fold{fold}" / f"seed{seed}"
        run.mkdir(parents=True)
        metadata = {
            "experiment": name,
            "model_type": "dmpnn",
            "fold": fold,
            "seed": seed,
            "status": "completed",
            "start_time": f"2026-01-0{fold + 1}T00:00:00+00:00",
            "end_time": f"2026-01-0{fold + 1}T00:10:00+00:00",
            "runtime_seconds": 600,
            "best_epoch": 7,
            "selection_metric": "macro_ST_RAE",
            "final_metrics": {
                "macro_ST_RAE": st_rae,
                "macro_MAE": 0.5,
                "macro_RMSE": 0.7,
                "macro_Spearman": 0.6,
                "macro_R2": 0.3,
            },
            "model_hyperparameters": {"depth": 3},
            "training_hyperparameters": {"batch_size": 64},
        }
        (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (run / "config.yaml").write_text(
            "data:\n  split_scheme: ecfp_cluster\nloss:\n  loss_mode: standard\n",
            encoding="utf-8",
        )
        pd.DataFrame(
            [{
                "CYP": "CYP2D6", "N": 10, "ST_RAE": st_rae, "MAE": 0.5,
                "RMSE": 0.7, "Spearman": 0.6, "R2": 0.3,
            }]
        ).to_csv(run / "metrics_per_cyp.csv", index=False)
        return run

    def test_registry_upserts_runs_and_preserves_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "experiment_leaderboard.csv"
            first_run = self.write_run(root, "transfer", 0, 1, 0.7)
            first = update_model_statistics(root, output)
            self.assertEqual(list(first.columns), MODEL_STATISTICS_COLUMNS)
            self.assertEqual(len(first), 1)
            registered = first.loc[0, "registered_at_utc"]
            metrics = pd.read_csv(first_run / "metrics_per_cyp.csv")
            metrics.loc[0, "ST_RAE"] = 0.65
            metrics.to_csv(first_run / "metrics_per_cyp.csv", index=False)
            self.write_run(root, "transfer", 1, 2, 0.6)
            second = update_model_statistics(root, output)
            self.assertEqual(len(second), 2)
            original = second.loc[second["fold"].eq(0)].iloc[0]
            self.assertEqual(original["registered_at_utc"], registered)
            self.assertAlmostEqual(original["ST_RAE"], 0.65)
            self.assertEqual(set(second["CV_runs"]), {2})
            self.assertAlmostEqual(float(second["CV_ST_RAE_mean"].iloc[0]), 0.625)

    def test_registry_migrates_legacy_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "experiment_leaderboard.csv"
            pd.DataFrame([{"experiment": "legacy"}]).to_csv(output, index=False)
            self.write_run(root, "new-model", 0, 1, 0.7)
            result = update_model_statistics(root, output)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.loc[0, "experiment"], "new-model")

    def test_registry_retains_rows_for_archived_run_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "experiment_leaderboard.csv"
            run = self.write_run(root, "archived", 0, 1, 0.7)
            first = update_model_statistics(root, output)
            archived_run_id = first.loc[0, "run_id"]
            archived_registered_at = first.loc[0, "registered_at_utc"]
            for path in sorted(run.glob("*")):
                path.unlink()
            run.rmdir()
            result = update_model_statistics(root, output)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.loc[0, "run_id"], archived_run_id)
            self.assertEqual(result.loc[0, "registered_at_utc"], archived_registered_at)

    def test_registry_forward_migrates_and_retains_archived_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "experiment_leaderboard.csv"
            archived = {
                "run_id": "archived/fold0/seed1",
                "experiment": "archived",
                "model": "dmpnn",
                "CYP": "CYP2D6",
                "registered_at_utc": "2026-01-01T00:00:00+00:00",
                "completion_status": "completed",
            }
            # Simulate a valid older schema that predates model_id and capacity fields.
            pd.DataFrame([archived]).to_csv(output, index=False)
            self.write_run(root, "new-model", 0, 1, 0.7)
            result = update_model_statistics(root, output)
            self.assertEqual(set(result["run_id"]), {archived["run_id"], "new-model/fold0/seed1"})
            old = result.loc[result["run_id"].eq(archived["run_id"])].iloc[0]
            self.assertEqual(old["registered_at_utc"], archived["registered_at_utc"])
            self.assertTrue(pd.isna(old["model_id"]))

    def test_completed_run_updates_registry_automatically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.write_run(root, "automatic", 0, 1, 0.7)
            metadata_path = run / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["status"] = "running"
            metadata["output_root"] = str(root)
            context = RunContext(run, resume=True)
            finish_run(context, metadata, time.monotonic(), {"final_metrics": metadata["final_metrics"]})
            registry = pd.read_csv(root / "experiment_leaderboard.csv")
            self.assertEqual(len(registry), 1)
            self.assertEqual(registry.loc[0, "completion_status"], "completed")
            self.assertTrue((run / "COMPLETED").exists())


if __name__ == "__main__":
    unittest.main()
