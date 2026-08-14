# Copyright (c) 2026
# All rights reserved.

import importlib.util
import unittest
from pathlib import Path

ATLAS_ROOT = Path(__file__).resolve().parents[1]
RUN_PATH = ATLAS_ROOT / "sim/matmul/run.py"
SPEC = importlib.util.spec_from_file_location("atlas_matmul_model", RUN_PATH)
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)
compare_cycles = RUN.compare_cycles
summarize_profile = RUN.summarize_profile
validate_shape = RUN.validate_shape


class MatmulModelTest(unittest.TestCase):
    def test_aligned_shape_matches_naive_compute_cycles(self):
        result = compare_cycles(256, 1024, 128, 64, 12000, 33554432)
        self.assertEqual(result["mac_count"], 33554432)
        self.assertEqual(result["mac_capacity_per_cycle"], 4096)
        self.assertEqual(result["naive_cycles"], 8192)
        self.assertEqual(result["gem5_compute_cycles"], 8192)
        self.assertEqual(result["compute_over_naive"], 1.0)

    def test_partial_tiles_expose_utilization_loss(self):
        result = compare_cycles(65, 65, 32, 64, 200, 135200)
        self.assertEqual(result["naive_cycles"], 34)
        self.assertEqual(result["gem5_compute_cycles"], 128)
        self.assertGreater(result["compute_over_naive"], 3.7)

    def test_invalid_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            validate_shape(0, 1, 16)

    def test_profile_ticks_are_converted_to_cycles(self):
        profile = {
            "axis_count": 2,
            "event_count": 2,
            "events": [
                {
                    "axis": "system.mpu",
                    "start_tick": 1000,
                    "end_tick": 5000,
                },
                {
                    "axis": "system.mpu.uops",
                    "start_tick": 1500,
                    "end_tick": 4500,
                },
            ],
        }
        summary = summarize_profile(profile, clock_mhz=2000)
        self.assertEqual(summary["macro_event_count"], 1)
        self.assertEqual(summary["uop_event_count"], 1)
        self.assertEqual(summary["span_ticks"], 4000)
        self.assertEqual(summary["span_cycles"], 8)


if __name__ == "__main__":
    unittest.main()
