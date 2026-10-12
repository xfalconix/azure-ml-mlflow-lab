"""Check the evaluation boundaries that protect this small experiment."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train import build_candidates, select_winner, split_data


class TrainingTests(unittest.TestCase):
    def test_partitions_are_disjoint_and_cover_every_row(self):
        features, _, split = split_data()
        train, validation, test = (set(split[name]) for name in ["train", "validation", "test"])
        self.assertFalse(train & validation or train & test or validation & test)
        self.assertEqual(train | validation | test, set(features.index))
        self.assertEqual([len(train), len(validation), len(test)], [264, 89, 89])

    def test_selection_ignores_test_results(self):
        comparison = [
            {"model": "ridge", "validation_mae": 40, "test_mae": 100},
            {"model": "random_forest", "validation_mae": 45, "test_mae": 1},
        ]
        self.assertEqual(select_winner(comparison), "ridge")

    def test_scaler_learns_training_rows_only(self):
        features, target, split = split_data()
        ridge = build_candidates()["ridge"]
        ridge.fit(features.loc[split["train"]], target.loc[split["train"]])
        np.testing.assert_allclose(
            ridge.named_steps["standardscaler"].mean_, features.loc[split["train"]].mean()
        )


if __name__ == "__main__":
    unittest.main()
