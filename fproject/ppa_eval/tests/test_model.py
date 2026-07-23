# Copyright (c) 2026
# All rights reserved.

import tempfile
import unittest
from pathlib import Path

from fproject.ppa_eval.model import (
    PPARecord,
    compare_results,
    evaluate_config,
    load_config,
    select_record,
)


def make_record(
    record_id,
    design,
    *,
    pdk="T7",
    variant="default",
    slack=0.0,
    timing_met=True,
    power=1.0,
    area=1.0,
):
    return PPARecord(
        record_id=record_id,
        source_kind="source",
        pdk=pdk,
        flow="DC-Innovus",
        design=design,
        variant=variant,
        utilization_pct=50.0,
        target_period_ns=0.5,
        timing_slack_ns=slack,
        timing_met=timing_met,
        area_um2=area * 1_000_000,
        area_mm2=area,
        power_w=power,
        source_row="2",
        derived_from="",
        power_scale=1.0,
        area_scale=1.0,
    )


class ModelTest(unittest.TestCase):
    def test_timing_fallback_selects_greatest_slack(self):
        records = [
            make_record(
                "worse",
                "Mesh_BOTH_32x32",
                slack=-0.20,
                timing_met=False,
            ),
            make_record(
                "better",
                "Mesh_BOTH_32x32",
                slack=-0.012,
                timing_met=False,
            ),
        ]
        selected, details = select_record(
            records,
            pdk="T7",
            flow="DC-Innovus",
            design="Mesh_BOTH_32x32",
            utilization_pct=50,
            system_period_ns=0.5,
        )
        self.assertEqual(selected.record_id, "better")
        self.assertTrue(details["timing_fallback"])
        self.assertIn("greatest timing slack", details["warning"])

    def test_config_defaults(self):
        content = """
schema_version: 1
name: sample
system:
  frequency_mhz: 2000
  pdk: T7
compute:
  vector:
    fp32_ops_per_cycle: 128
  tensor:
    array_dim: 128
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(content, encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config["system"]["flow"], "DC-Innovus")
        self.assertEqual(
            config["compute"]["vector"]["implementation_variant"],
            "no_macro",
        )
        self.assertEqual(config["compute"]["tensor"]["utilization_pct"], 50.0)

    def test_config_allows_simulation_section_without_affecting_ppa(self):
        content = """
schema_version: 1
name: sample
system:
  frequency_mhz: 2000
  pdk: T7
compute:
  vector:
    fp32_ops_per_cycle: 128
  tensor:
    array_dim: 128
simulation:
  clock_mhz: 1000
  spm:
    size_bytes: 8388608
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(content, encoding="utf-8")
            config = load_config(path)
        self.assertNotIn("simulation", config)
        self.assertEqual(config["system"]["frequency_mhz"], 2000.0)

    def test_evaluate_counts_and_totals(self):
        records = [
            make_record(
                "vector",
                "ara_sys",
                variant="no_macro",
                slack=-0.27,
                timing_met=False,
                power=1.0,
                area=0.5,
            ),
            make_record(
                "tensor",
                "Mesh_BOTH_32x32",
                slack=-0.012,
                timing_met=False,
                power=2.0,
                area=0.25,
            ),
        ]
        config = {
            "name": "T7",
            "system": {
                "frequency_mhz": 2000.0,
                "pdk": "T7",
                "flow": "DC-Innovus",
            },
            "compute": {
                "vector": {
                    "fp32_ops_per_cycle": 128.0,
                    "utilization_pct": 50.0,
                    "implementation_variant": "no_macro",
                },
                "tensor": {
                    "array_dim": 128.0,
                    "utilization_pct": 50.0,
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            datasheet = Path(directory) / "data.csv"
            datasheet.write_text("fixture", encoding="utf-8")
            result = evaluate_config(config, records, datasheet)
        self.assertEqual(result["modules"]["vector"]["instance_count"], 32)
        self.assertEqual(result["modules"]["tensor"]["instance_count"], 16)
        self.assertEqual(result["totals"]["power_w"], 64)
        self.assertEqual(result["totals"]["area_mm2"], 20)
        self.assertEqual(len(result["warnings"]), 2)

    def test_comparison_improvement(self):
        baseline = {"totals": {"power_w": 10.0, "area_mm2": 4.0}}
        candidate = {"totals": {"power_w": 8.0, "area_mm2": 3.0}}
        comparison = compare_results(baseline, candidate)["comparison"]
        self.assertAlmostEqual(comparison["power_improvement_pct"], 20.0)
        self.assertAlmostEqual(comparison["area_improvement_pct"], 25.0)


if __name__ == "__main__":
    unittest.main()
