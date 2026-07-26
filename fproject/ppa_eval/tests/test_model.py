# Copyright (c) 2026
# All rights reserved.

import tempfile
import unittest
from pathlib import Path

from fproject.ppa_eval.model import (
    PPARecord,
    SRAMRecord,
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
    shared_power=0.2,
    shared_area=0.1,
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
        shared_power_w=shared_power,
        shared_area_um2=shared_area * 1_000_000,
        shared_area_mm2=shared_area,
        shared_derived_from="fixture-cpu,fixture-other",
    )


def make_sram_record(
    *,
    pdk="T7",
    power_mw=0.063735,
    area_um2=23249.16,
):
    return SRAMRecord(
        record_id=f"sram-{pdk}",
        source_kind="spec",
        pdk=pdk,
        design="SRAM_16384x32",
        depth=16384,
        word_bits=32,
        capacity_bytes=65536,
        max_frequency_mhz=2000.0,
        static_power_mw=power_mw,
        static_power_w=power_mw / 1000.0,
        area_um2=area_um2,
        area_mm2=area_um2 / 1_000_000.0,
        derived_from="",
        static_power_scale=1.0,
        area_scale=1.0,
        provenance="fixture",
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
    fp32_elements_per_cycle: 128
  tensor:
    array_dim: 64
memory:
  sram:
    capacity_bytes: 262144
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(content, encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config["system"]["flow"], "DC-Innovus")
        self.assertEqual(
            config["compute"]["vector"]["implementation_variant"],
            "with_macro",
        )
        self.assertEqual(config["compute"]["tensor"]["utilization_pct"], 50.0)
        self.assertEqual(config["memory"]["sram"]["capacity_bytes"], 262144)

    def test_config_allows_simulation_section_without_affecting_ppa(self):
        content = """
schema_version: 1
name: sample
system:
  frequency_mhz: 2000
  pdk: T7
compute:
  vector:
    fp32_elements_per_cycle: 128
  tensor:
    array_dim: 64
memory:
  sram:
    capacity_bytes: 262144
simulation:
  clock_mhz: 2000
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
                    "fp32_elements_per_cycle": 128.0,
                    "utilization_pct": 50.0,
                    "implementation_variant": "no_macro",
                },
                "tensor": {
                    "array_dim": 64.0,
                    "utilization_pct": 50.0,
                },
            },
            "memory": {"sram": {"capacity_bytes": 262144}},
        }
        with tempfile.TemporaryDirectory() as directory:
            datasheet = Path(directory) / "data.csv"
            datasheet.write_text("fixture", encoding="utf-8")
            sram_datasheet = Path(directory) / "sram.csv"
            sram_datasheet.write_text("fixture", encoding="utf-8")
            result = evaluate_config(
                config,
                records,
                datasheet,
                [make_sram_record()],
                sram_datasheet,
            )
        self.assertEqual(result["modules"]["vector"]["instance_count"], 16)
        self.assertEqual(result["modules"]["tensor"]["instance_count"], 4)
        self.assertEqual(result["modules"]["sram"]["instance_count"], 4)
        self.assertEqual(
            result["modules"]["vector"]["requested_ops_per_cycle"], 256
        )
        self.assertEqual(
            result["modules"]["tensor"]["requested_ops_per_cycle"], 8192
        )
        self.assertAlmostEqual(result["totals"]["compute_power_w"], 21)
        self.assertAlmostEqual(
            result["totals"]["sram_static_power_w"], 0.00025494
        )
        self.assertAlmostEqual(result["totals"]["power_w"], 21.00025494)
        self.assertAlmostEqual(result["totals"]["compute_area_mm2"], 7.5)
        self.assertAlmostEqual(result["totals"]["area_mm2"], 7.59299664)
        self.assertEqual(result["modules"]["vector"]["shared_instance_count"], 1)
        self.assertAlmostEqual(
            result["modules"]["vector"]["lane_group_power_w"], 0.8
        )
        self.assertEqual(len(result["warnings"]), 2)

    def test_sram_capacity_rounds_up_to_whole_macros(self):
        records = [
            make_record("vector", "ara_sys", variant="no_macro"),
            make_record("tensor", "Mesh_BOTH_32x32"),
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
                    "fp32_elements_per_cycle": 8.0,
                    "utilization_pct": 50.0,
                    "implementation_variant": "no_macro",
                },
                "tensor": {
                    "array_dim": 32.0,
                    "utilization_pct": 50.0,
                },
            },
            "memory": {"sram": {"capacity_bytes": 65537}},
        }
        with tempfile.TemporaryDirectory() as directory:
            datasheet = Path(directory) / "data.csv"
            datasheet.write_text("fixture", encoding="utf-8")
            sram_datasheet = Path(directory) / "sram.csv"
            sram_datasheet.write_text("fixture", encoding="utf-8")
            result = evaluate_config(
                config,
                records,
                datasheet,
                [make_sram_record()],
                sram_datasheet,
            )
        self.assertEqual(result["modules"]["sram"]["instance_count"], 2)
        self.assertEqual(
            result["modules"]["sram"]["provisioned_capacity_bytes"], 131072
        )

    def test_sram_rejects_frequency_above_two_ghz(self):
        config = {
            "name": "T7",
            "system": {
                "frequency_mhz": 2001.0,
                "pdk": "T7",
                "flow": "DC-Innovus",
            },
            "compute": {
                "vector": {
                    "fp32_elements_per_cycle": 8.0,
                    "utilization_pct": 50.0,
                    "implementation_variant": "no_macro",
                },
                "tensor": {
                    "array_dim": 32.0,
                    "utilization_pct": 50.0,
                },
            },
            "memory": {"sram": {"capacity_bytes": 65536}},
        }
        records = [
            make_record("vector", "ara_sys", variant="no_macro"),
            make_record("tensor", "Mesh_BOTH_32x32"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            datasheet = Path(directory) / "data.csv"
            datasheet.write_text("fixture", encoding="utf-8")
            sram_datasheet = Path(directory) / "sram.csv"
            sram_datasheet.write_text("fixture", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at most 2000 MHz"):
                evaluate_config(
                    config,
                    records,
                    datasheet,
                    [make_sram_record()],
                    sram_datasheet,
                )

    def test_comparison_improvement(self):
        baseline = {"totals": {"power_w": 10.0, "area_mm2": 4.0}}
        candidate = {"totals": {"power_w": 8.0, "area_mm2": 3.0}}
        comparison = compare_results(baseline, candidate)["comparison"]
        self.assertAlmostEqual(comparison["power_improvement_pct"], 20.0)
        self.assertAlmostEqual(comparison["area_improvement_pct"], 25.0)


if __name__ == "__main__":
    unittest.main()
