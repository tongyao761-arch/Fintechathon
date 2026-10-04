"""The reporting shim uses actual frozen evidence and never invents results."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from artifacts.features34_step6 import audit_compat as compat


class AuditCompatibilityTests(unittest.TestCase):
    def test_baseline_only_uses_exact_existing_frozen_entry(self):
        base = ('baseline10', 'primary_2023'); candidate = ('S4R_lean31_minus4', 'primary_2023')
        entries = {base: ('actual_baseline_directory', {'metrics': 1})}
        frozen = ('actual_frozen_directory', {'metrics': 2})
        with patch.object(compat, 'REFERENCES', {candidate: frozen}), patch.object(compat, 'ORIGINAL_TABLES', return_value='table') as build:
            self.assertEqual(compat.complete_tables(entries, {}), 'table')
            self.assertEqual(build.call_args.args[0], {base: entries[base], candidate: frozen})
            self.assertEqual(set(entries), {base})

    def test_missing_frozen_comparator_is_rejected(self):
        with patch.object(compat, 'REFERENCES', {}), patch.object(compat, 'ORIGINAL_TABLES') as build:
            with self.assertRaises(AssertionError): compat.complete_tables({('baseline10', 'primary_2023'): (None, None)}, {})
            build.assert_not_called()


if __name__ == '__main__': unittest.main()
