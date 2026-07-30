# Copyright (c) 2026
# All rights reserved.

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOL_ROOT = ROOT / "fproject" / "ppa_eval"


class CliTest(unittest.TestCase):
    def test_t7_f7_comparison_json(self):
        result = subprocess.run(
            [
                sys.executable,
                str(TOOL_ROOT / "ppa_eval.py"),
                str(TOOL_ROOT / "configs" / "t7.yaml"),
                str(TOOL_ROOT / "configs" / "f7.yaml"),
                "--format",
                "json",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(
            report["baseline"]["modules"]["vector"]["instance_count"], 16
        )
        self.assertEqual(
            report["candidate"]["modules"]["tensor"]["instance_count"], 4
        )
        sram = report["baseline"]["modules"]["sram"]
        self.assertEqual(sram["capacity_instance_count"], 4)
        self.assertEqual(sram["bandwidth_instance_count"], 32)
        self.assertEqual(sram["instance_count"], 32)
        self.assertEqual(sram["bank_count"], 32)
        self.assertEqual(sram["bank_width_bytes"], 4)
        self.assertEqual(sram["required_bytes_per_cycle"], 256)
        self.assertEqual(sram["required_read_bytes_per_cycle"], 128)
        self.assertEqual(sram["required_write_bytes_per_cycle"], 128)
        self.assertTrue(sram["simultaneous_read_write"])
        self.assertAlmostEqual(
            report["baseline"]["totals"]["sram_power_w"], 1.33088
        )
        self.assertTrue(
            report["baseline"]["modules"]["tensor"]["selection"][
                "timing_fallback"
            ]
        )

    def test_run_script_prints_acceptance_summary(self):
        result = subprocess.run(
            [str(TOOL_ROOT / "run.sh")],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PPA COMPARISON: T7 (baseline) -> F7", result.stdout)
        self.assertIn("Power (W)", result.stdout)
        self.assertIn("Area (mm^2)", result.stdout)
        self.assertIn(
            "1 x shared CPU/other + 16 x 4-lane Vector/SRAM",
            result.stdout,
        )
        self.assertIn("record=source-sheet1-r32", result.stdout)
        self.assertIn("timing_met=True", result.stdout)
        self.assertIn("peak=16.384000 TOPS", result.stdout)
        self.assertIn("requested=256.000 KiB", result.stdout)
        self.assertIn("capacity_instances=4", result.stdout)
        self.assertIn("bandwidth_instances=32", result.stdout)
        self.assertIn("banks=32x4B", result.stdout)
        self.assertIn("simultaneous_rw=True", result.stdout)
        self.assertIn("bandwidth=256 B/cycle", result.stdout)

    def test_vector2x_configuration_scales_vector_and_sram(self):
        result = subprocess.run(
            [
                sys.executable,
                str(TOOL_ROOT / "ppa_eval.py"),
                str(
                    TOOL_ROOT
                    / "configs"
                    / "f7_vector2x_double_buffer.yaml"
                ),
                "--format",
                "json",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["modules"]["vector"]["instance_count"], 32)
        self.assertEqual(report["modules"]["sram"]["instance_count"], 96)
        self.assertAlmostEqual(report["totals"]["power_w"], 23.52232)
        self.assertAlmostEqual(report["totals"]["area_mm2"], 3.9383025505)


if __name__ == "__main__":
    unittest.main()
