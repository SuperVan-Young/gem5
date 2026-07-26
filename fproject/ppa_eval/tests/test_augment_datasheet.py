# Copyright (c) 2026
# All rights reserved.

import unittest

from fproject.ppa_eval.augment_datasheet import augment_f7_ara


class AugmentDatasheetTest(unittest.TestCase):
    def setUp(self):
        self.source = {
            "record_id": "source-sheet1-r30",
            "source_kind": "source",
            "pdk": "T7",
            "raw_pdk": "T7",
            "raw_flow": "T7-DC-Innovus",
            "design": "ara_sys",
            "area_um2": "400000",
            "area_mm2": "0.4",
            "power_w": "1.2",
            "shared_power_w": "0.2",
            "shared_area_um2": "100000",
            "shared_area_mm2": "0.1",
            "shared_derived_from": "source-sheet1-r27,source-sheet1-r29",
            "notes": "no macro",
        }

    def test_scales_power_area_and_preserves_provenance(self):
        result = augment_f7_ara([self.source])
        augmented = result[1]
        self.assertEqual(augmented["pdk"], "F7-PKU")
        self.assertEqual(augmented["source_kind"], "augmented")
        self.assertEqual(augmented["derived_from"], "source-sheet1-r30")
        self.assertAlmostEqual(augmented["power_w"], 0.6)
        self.assertAlmostEqual(augmented["area_um2"], 300000)
        self.assertAlmostEqual(augmented["area_mm2"], 0.3)
        self.assertAlmostEqual(augmented["shared_power_w"], 0.1)
        self.assertAlmostEqual(augmented["shared_area_um2"], 75000)

    def test_refuses_to_overwrite_real_target_data(self):
        existing = dict(self.source, pdk="F7-PKU")
        with self.assertRaisesRegex(ValueError, "already contains"):
            augment_f7_ara([self.source, existing])


if __name__ == "__main__":
    unittest.main()
