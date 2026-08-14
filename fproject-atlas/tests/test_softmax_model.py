# Copyright (c) 2026
# All rights reserved.

import importlib.util
import unittest
from pathlib import Path

ATLAS_ROOT = Path(__file__).resolve().parents[1]
RUN_PATH = ATLAS_ROOT / "sim/softmax/run.py"
SPEC = importlib.util.spec_from_file_location("atlas_softmax_model", RUN_PATH)
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)
compare_cycles = RUN.compare_cycles
summarize_profile = RUN.summarize_profile
validate_shape = RUN.validate_shape


class SoftmaxModelTest(unittest.TestCase):
    def setUp(self):
        self.vector = {
            "dlen_bytes": 512,
            "float32_cycles_per_dlen": 1,
        }

    def test_naive_vector_pass_model(self):
        result = compare_cycles(256, 1024, self.vector, 17648)
        self.assertEqual(result["atlas_ops_per_element"], 9)
        self.assertEqual(result["atlas_vec_count"], 2359296)
        self.assertEqual(result["vector_vec_num"], 128)
        self.assertEqual(result["naive_cycles"], 18432)
        self.assertEqual(result["gem5_busy_cycles"], 17648)

    def test_layout_validation(self):
        validate_shape(256, 1024, 512)
        with self.assertRaisesRegex(ValueError, "multiples of 128"):
            validate_shape(64, 1024, 512)

    def test_profile_busy_time_sums_macros(self):
        profile = {
            "axis_count": 2,
            "event_count": 3,
            "events": [
                {
                    "axis": "system.vpu0",
                    "start_tick": 1000,
                    "end_tick": 3000,
                    "duration": 2000,
                    "begin": {"opcode": 11},
                },
                {
                    "axis": "system.vpu0",
                    "start_tick": 3500,
                    "end_tick": 5000,
                    "duration": 1500,
                    "begin": {"opcode": 14},
                },
                {
                    "axis": "system.vpu0.uops",
                    "start_tick": 1200,
                    "end_tick": 2800,
                    "duration": 1600,
                    "begin": {"opcode": 11},
                },
            ],
        }
        summary = summarize_profile(profile, clock_mhz=2000)
        self.assertEqual(summary["macro_event_count"], 2)
        self.assertEqual(summary["busy_cycles"], 7)
        self.assertEqual(summary["span_cycles"], 8)
        self.assertEqual(summary["opcode_counts"]["reduce_max"], 1)
        self.assertEqual(summary["opcode_counts"]["exp"], 1)
        self.assertEqual(summary["opcode_busy_cycles"]["reduce_max"], 4)
        self.assertEqual(summary["opcode_busy_cycles"]["exp"], 3)


if __name__ == "__main__":
    unittest.main()
