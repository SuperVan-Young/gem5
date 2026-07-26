# Copyright (c) 2026
# All rights reserved.

import csv
import tempfile
import unittest
from pathlib import Path

from fproject.ppa_eval.model import load_sram_records

ROOT = Path(__file__).resolve().parents[3]
SRAM_DATASHEET = ROOT / "fproject/ppa_eval/data/sram_v1.csv"


class SRAMDataTest(unittest.TestCase):
    def test_checked_in_t7_and_f7_records(self):
        records = load_sram_records(SRAM_DATASHEET)
        by_pdk = {record.pdk: record for record in records}
        t7 = by_pdk["T7"]
        f7 = by_pdk["F7-PKU"]
        self.assertEqual(t7.design, "SRAM_4096x128")
        self.assertEqual(t7.depth, 4096)
        self.assertEqual(t7.word_bits, 128)
        self.assertEqual(t7.capacity_bytes, 65536)
        self.assertEqual(t7.depth * t7.word_bits // 8, 65536)
        self.assertAlmostEqual(t7.power_w, 0.041590)
        self.assertAlmostEqual(t7.area_um2, 20923.128)
        self.assertEqual(t7.power_scale, 1.0)
        self.assertEqual(t7.area_scale, 1.0)
        self.assertEqual(f7.derived_from, t7.record_id)
        self.assertEqual(f7.power_scale, 0.5)
        self.assertEqual(f7.area_scale, 0.5)
        self.assertAlmostEqual(f7.power_w, t7.power_w * 0.5)
        self.assertAlmostEqual(f7.area_um2, t7.area_um2 * 0.5)

    def test_rejects_inconsistent_capacity(self):
        with SRAM_DATASHEET.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
            fields = reader.fieldnames
        rows[0]["capacity_bytes"] = "1"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sram.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "organization"):
                load_sram_records(path)


if __name__ == "__main__":
    unittest.main()
