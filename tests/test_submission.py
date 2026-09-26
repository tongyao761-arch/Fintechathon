import unittest

import pandas as pd

from src.data.data_contract import DataContractError
from src.validation.submission import validate_submission


class TestSubmission(unittest.TestCase):
    def test_accepts_valid_keys_in_a_different_order(self):
        test = pd.DataFrame({"ts_code": ["a", "b"], "trade_date": [20240101, 20240101], "row_id": [0, 1]})
        prediction = pd.DataFrame({"ts_code": ["b", "a"], "trade_date": [20240101, 20240101], "pred": [0.2, 0.1]})
        self.assertEqual(validate_submission(prediction, test)["rows"], 2)

    def test_validates_key_set_not_only_row_count(self):
        test = pd.DataFrame({"ts_code": ["a", "b"], "trade_date": [20240101, 20240101], "row_id": [0, 1]})
        bad = pd.DataFrame({"ts_code": ["a", "c"], "trade_date": [20240101, 20240101], "pred": [0.1, 0.2]})
        with self.assertRaises(DataContractError):
            validate_submission(bad, test)


if __name__ == "__main__":
    unittest.main()
