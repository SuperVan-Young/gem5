# Copyright (c) 2026
# All rights reserved.

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fproject.ppa_eval.normalize_snapshot import (
    normalize_snapshot,
    parse_timing,
)


class NormalizeSnapshotTest(unittest.TestCase):
    def test_parse_timing_period_and_slack(self):
        self.assertEqual(parse_timing("0.500/-0.012", 0.5), (0.5, -0.012))
        self.assertEqual(parse_timing("0.18", 1.0), (1.0, 0.18))

    def test_normalizes_area_flow_variant_and_negative_zero(self):
        cells = {
            1: "T7",
            2: "T7-DC-Innovus",
            3: "ara_sys",
            4: "0.5",
            5: "0.5",
            8: "500000",
            10: "0.500/-0.0",
            11: "1.25",
            13: "no macro, all innovus",
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.xlsx"
            source.write_bytes(b"fixture")
            with mock.patch(
                "fproject.ppa_eval.normalize_snapshot.read_first_sheet_rows",
                return_value=[(2, cells)],
            ):
                records, ignored = normalize_snapshot(source)

        self.assertFalse(ignored)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["flow"], "DC-Innovus")
        self.assertEqual(record["variant"], "no_macro")
        self.assertTrue(record["timing_met"])
        self.assertEqual(record["area_mm2"], 0.5)


if __name__ == "__main__":
    unittest.main()
