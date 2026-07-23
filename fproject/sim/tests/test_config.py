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
    def test_f7_hardware_has_separate_simulation_clock(self):
        hardware = load_hardware(ROOT / "fproject/ppa_eval/configs/f7.yaml")
        self.assertEqual(hardware["ppa_frequency_mhz"], 2000)
        self.assertEqual(hardware["simulation"]["clock_mhz"], 1000)
        self.assertEqual(hardware["array_dim"], 128)

    def test_runtime_shapes_are_loaded_from_task_files(self):
        first = load_task(
            ROOT / "fproject/sim/configs/fa_q256_kv1024_d128.yaml"
        )
        second = load_task(
            ROOT / "fproject/sim/configs/fa_q128_kv512_d128.yaml"
        )
        self.assertEqual(
            (first["q"], first["kv"], first["d"]), (256, 1024, 128)
        )
        self.assertEqual(
            (second["q"], second["kv"], second["d"]), (128, 512, 128)
        )

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
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.yaml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_task(path)


if __name__ == "__main__":
    unittest.main()
