# Copyright (c) 2026
# All rights reserved.

import tempfile
import unittest
from pathlib import Path

from fproject.sim.fa_sim import (
    ConfigError,
    load_hardware,
    load_task,
)

ROOT = Path(__file__).resolve().parents[3]


class ConfigTest(unittest.TestCase):
    def test_f7_hardware_uses_ppa_compute_rate(self):
        hardware = load_hardware(ROOT / "fproject/ppa_eval/configs/f7.yaml")
        self.assertEqual(hardware["ppa_frequency_mhz"], 2000)
        self.assertEqual(hardware["simulation"]["clock_mhz"], 2000)
        self.assertEqual(hardware["array_dim"], 64)
        self.assertEqual(hardware["tensor_ops_per_cycle"], 8192)
        self.assertEqual(hardware["vector_fp32_elements_per_cycle"], 128)
        self.assertEqual(hardware["vector_fp32_flops_per_cycle"], 256)
        self.assertEqual(hardware["ppa_sram_capacity_bytes"], 262144)
        self.assertEqual(
            hardware["buffer_pipeline"],
            {
                "bank_count": 32,
                "bank_width_bytes": 4,
                "banks_per_engine": 32,
                "buffer_slots": 1,
                "context_stride_bytes": 262144,
                "split_dimension": "br",
            },
        )
        self.assertEqual(
            hardware["simulation"]["spm"]["size_bytes"], 8 * 1024 * 1024
        )

    def test_rejects_simulation_clock_that_differs_from_ppa(self):
        source = ROOT / "fproject/ppa_eval/configs/f7.yaml"
        content = source.read_text(encoding="utf-8").replace(
            "clock_mhz: 2000", "clock_mhz: 1000"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hardware.yaml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "must match"):
                load_hardware(path)

    def test_runtime_shapes_are_loaded_from_task_files(self):
        first = load_task(
            ROOT / "fproject/sim/configs/fa_q256_kv1024_d128.yaml"
        )
        second = load_task(
            ROOT / "fproject/sim/configs/fa_q128_kv512_d128.yaml"
        )
        edge = load_task(ROOT / "fproject/sim/configs/fa_q192_kv512_d128.yaml")
        self.assertEqual(
            (first["q"], first["kv"], first["d"]), (256, 1024, 128)
        )
        self.assertEqual(
            (second["q"], second["kv"], second["d"]), (128, 512, 128)
        )
        self.assertEqual((first["br"], first["bc"]), (128, 128))
        self.assertEqual((second["br"], second["bc"]), (128, 128))
        self.assertEqual((edge["q"], edge["kv"]), (192, 512))
        self.assertEqual((edge["br"], edge["bc"]), (128, 128))

    def test_rejects_non_positive_shape(self):
        content = """
schema_version: 1
name: invalid
operator: flash_attention_v2
attention:
  q: 0
  kv: 512
  d: 128
  batch: 1
  heads: 1
  input_dtype: int8
  accumulation_dtype: int32
softmax_dtype: float32
tiling:
  q: 128
  kv: 128
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.yaml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_task(path)

    def test_rejects_non_128_tiling(self):
        content = """
schema_version: 1
name: invalid_tiling
operator: flash_attention_v2
attention:
  q: 128
  kv: 512
  d: 128
  batch: 1
  heads: 1
  input_dtype: int8
  accumulation_dtype: int32
  softmax_dtype: float32
tiling:
  q: 64
  kv: 128
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.yaml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "both be 128"):
                load_task(path)


if __name__ == "__main__":
    unittest.main()
