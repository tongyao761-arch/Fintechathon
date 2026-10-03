import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import run_features34 as runner
from src.features.features34 import build_features34
from tests.test_features34 import panel


class TestStep2Authorization(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((runner.ROOT / "configs/features34_step2.json").read_text())

    def test_all_fourteen_inputs_equal_full_calculation_subset(self):
        p = panel()
        full = build_features34(p)
        self.assertEqual(len(self.config["experiments"]), 14)
        for name in self.config["experiments"]:
            columns = runner.resolve_selection(self.config, name, "primary_2023")
            pd.testing.assert_frame_equal(build_features34(p, columns), full.loc[:, columns], check_exact=True)

    def test_2024_rejected_even_if_config_allowed(self):
        for name in self.config["experiments"]:
            config = copy.deepcopy(self.config)
            config["allowed_runs"][name].append("oos_2024")
            with self.assertRaises(ValueError):
                runner.resolve_selection(config, name, "oos_2024")

    def test_modified_selection_and_model_fields_rejected(self):
        for change in ({"groups": ["A", "B"]}, {"groups": ["A"], "exclude": ["ret_2d"]},
                       {"groups": ["A"], "n_estimators": 200}):
            config = copy.deepcopy(self.config)
            config["experiments"]["10+A"] = change
            with self.assertRaises(ValueError):
                runner.resolve_selection(config, "10+A", "primary_2023")

    def test_failure_before_data_loading_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, "EXPERIMENT_ROOT", Path(directory)), patch.object(runner, "load_raw_baseline_panel") as load:
                with self.assertRaises(ValueError):
                    runner.main(["--config", str(runner.ROOT / "configs/features34_step2.json"),
                                 "--candidate", "baseline10", "--split", "oos_2024"])
                load.assert_not_called()
            output = next(Path(directory).glob("*/*"))
            state = json.loads((output / "status.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertIn("elapsed_seconds", state)
            self.assertFalse((output / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
