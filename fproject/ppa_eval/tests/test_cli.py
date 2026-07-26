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
        self.assertEqual(
            report["baseline"]["modules"]["sram"]["instance_count"], 4
        )
        self.assertAlmostEqual(
            report["baseline"]["totals"]["sram_static_power_w"], 0.00025494
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
        self.assertIn("45.738%", result.stdout)
        self.assertIn("34.662%", result.stdout)
        self.assertIn(
            "1 x shared CPU/other + 16 x 4-lane Vector/SRAM",
            result.stdout,
        )
        self.assertIn("record=source-sheet1-r32", result.stdout)
        self.assertIn("timing_met=True", result.stdout)
        self.assertIn("peak=16.384000 TOPS", result.stdout)
        self.assertIn("requested=256.000 KiB", result.stdout)
        self.assertIn("sram_power=static-only", result.stdout)


if __name__ == "__main__":
    unittest.main()
