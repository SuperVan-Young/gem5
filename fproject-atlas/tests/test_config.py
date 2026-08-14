# Copyright (c) 2026
# All rights reserved.

import importlib.util
import tempfile
import unittest
from pathlib import Path

ATLAS_ROOT = Path(__file__).resolve().parents[1]
RUN_PATH = ATLAS_ROOT / "sim/matmul/run.py"
SPEC = importlib.util.spec_from_file_location("atlas_matmul_config", RUN_PATH)
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)
load_hardware = RUN.load_hardware


class HardwareConfigTest(unittest.TestCase):
    def test_example_config(self):
        config = load_hardware(ATLAS_ROOT / "configs/gem5_template.yaml")
        self.assertEqual(config["name"], "gem5-template")
        self.assertEqual(config["mpu"]["array_dim"], 64)
        self.assertEqual(config["vector"]["dlen_bytes"], 512)
        self.assertEqual(config["memory"]["spm"]["base_address"], 0x60000000)

    def test_extra_fields_are_left_to_consumers(self):
        text = (ATLAS_ROOT / "configs/gem5_template.yaml").read_text(
            encoding="utf-8"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extended.yaml"
            path.write_text(text + "unknown: true\n", encoding="utf-8")
            self.assertTrue(load_hardware(path)["unknown"])

    def test_schema_version_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("schema_version: 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_hardware(path)


if __name__ == "__main__":
    unittest.main()
