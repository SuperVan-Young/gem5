# Copyright (c) 2026
# All rights reserved.

import unittest

from fproject.sim.buffer_pipeline_model import evaluate_buffer_pipeline


class BufferPipelineModelTest(unittest.TestCase):
    def test_32_banks_serialize_tensor_and_vector(self):
        result = evaluate_buffer_pipeline(
            qk_busy_cycles=8896,
            softmax_busy_cycles=17648,
            pv_busy_cycles=8896,
            bank_count=32,
            bank_width_bytes=4,
            banks_per_engine=32,
            buffer_slots=1,
        )
        self.assertEqual(result["bandwidth_bytes_per_cycle"], 128)
        self.assertFalse(result["overlap_enabled"])
        self.assertEqual(result["non_overlap_cycles"], 35440)
        self.assertEqual(result["overlapped_cycles"], 35440)
        self.assertEqual(result["speedup"], 1.0)

    def test_64_banks_enable_two_buffer_overlap(self):
        result = evaluate_buffer_pipeline(
            qk_busy_cycles=8896,
            softmax_busy_cycles=17648,
            pv_busy_cycles=8896,
            bank_count=64,
            bank_width_bytes=4,
            banks_per_engine=32,
            buffer_slots=2,
        )
        self.assertEqual(result["bandwidth_bytes_per_cycle"], 256)
        self.assertTrue(result["overlap_enabled"])
        self.assertEqual(result["overlapped_cycles"], 26544)
        self.assertEqual(result["saved_cycles"], 8896)
        self.assertAlmostEqual(result["speedup"], 1.3351416516)
        self.assertEqual(
            [stage["cycles"] for stage in result["stages"]],
            [4448, 8824, 8824, 4448],
        )

    def test_rejects_incomplete_engine_bank_group(self):
        with self.assertRaisesRegex(ValueError, "complete engine bank group"):
            evaluate_buffer_pipeline(
                qk_busy_cycles=1,
                softmax_busy_cycles=1,
                pv_busy_cycles=1,
                bank_count=31,
                bank_width_bytes=4,
                banks_per_engine=32,
                buffer_slots=1,
            )


if __name__ == "__main__":
    unittest.main()
