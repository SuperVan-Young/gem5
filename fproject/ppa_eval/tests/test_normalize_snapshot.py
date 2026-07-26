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

    def test_attaches_cpu_and_other_as_shared_vector_overhead(self):
        ara = {
            1: "T7",
            2: "T7-DC-Innovus",
            3: "ara_sys",
            4: "0.5",
            5: "0.5",
            8: "500000",
            10: "0.500/0",
            11: "1.25",
            13: "with macro",
        }
        cpu = {3: "cpu", 8: "179226.3", 11: "0.066"}
        other = {3: "other", 8: "3041.41", 11: "0.0124"}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.xlsx"
            source.write_bytes(b"fixture")
            with mock.patch(
                "fproject.ppa_eval.normalize_snapshot.read_first_sheet_rows",
                return_value=[(26, ara), (27, cpu), (29, other)],
            ):
                records, _ = normalize_snapshot(source)

        record = records[0]
        self.assertAlmostEqual(record["shared_power_w"], 0.0784)
        self.assertAlmostEqual(record["shared_area_um2"], 182267.71)
        self.assertEqual(
            record["shared_derived_from"],
            "source-sheet1-r27,source-sheet1-r29",
        )


if __name__ == "__main__":
    unittest.main()
