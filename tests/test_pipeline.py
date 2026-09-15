from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data import load_direct_data, load_single_concentration_data
from src.metrics import official_st_rae, regression_metrics
from src.splits import fold_column


class DataAndMetricTests(unittest.TestCase):
    def test_existing_folds_and_fold_safe_auxiliary_data(self):
        direct = load_direct_data("ecfp_cluster", 0)
        self.assertEqual(len(direct.frame), 4905)
        self.assertFalse((direct.train_mask & direct.validation_mask).any())
        auxiliary = load_single_concentration_data("ecfp_cluster", 0)
        fold = fold_column("ecfp_cluster")
        outer_names = set(auxiliary.frame.loc[auxiliary.frame[fold].eq(0), "Molecule_Name"])
        used_names = set(auxiliary.frame.loc[auxiliary.train_mask | auxiliary.validation_mask, "Molecule_Name"])
        self.assertTrue(outer_names.isdisjoint(used_names))

    def test_single_task_masks_only_include_observed_labels(self):
        bundle = load_direct_data("ecfp_cluster", 0, ["CYP2D6"])
        used = bundle.train_mask | bundle.validation_mask
        self.assertTrue(bundle.frame.loc[used, bundle.target_columns[0]].notna().all())
        self.assertLess(int(used.sum()), len(bundle.frame))

    def test_official_metric_interval_behavior(self):
        y = np.asarray([4.0, 6.0])
        low = np.asarray([3.8, 5.8])
        high = np.asarray([4.2, 6.2])
        self.assertEqual(official_st_rae(y, [4.1, 5.9], low, high), 0.0)
        self.assertAlmostEqual(official_st_rae(y, [5.0, 5.0], low, high), 1.0)

    def test_metric_schema(self):
        frame = pd.DataFrame(
            {"CYP": ["CYP1A2"] * 3, "y_true": [4.0, 5.0, 6.0],
             "y_pred": [4.1, 5.0, 5.9], "lower": [3.9, 4.9, 5.9],
             "upper": [4.1, 5.1, 6.1]}
        )
        per_cyp, macro = regression_metrics(frame)
        self.assertEqual(per_cyp.loc[0, "ST_RAE"], 0.0)
        self.assertIn("macro_RMSE", macro)


if __name__ == "__main__":
    unittest.main()
