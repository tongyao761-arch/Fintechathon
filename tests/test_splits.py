import unittest

import pandas as pd

from src.validation.splits import get_split


class TestSplits(unittest.TestCase):
    def test_primary_split_purges_boundary_day(self):
        dates = pd.DataFrame({"trade_date": [20221229, 20221230, 20230103]})
        train, valid = get_split("primary_2023").masks(dates)
        self.assertEqual(train.tolist(), [True, False, False])
        self.assertEqual(valid.tolist(), [False, False, True])

    def test_oos_is_strictly_after_training(self):
        split = get_split("oos_2024")
        self.assertLess(split.train_end, split.valid_start)
        self.assertNotEqual(split.purge_date, split.train_end)


if __name__ == "__main__":
    unittest.main()
