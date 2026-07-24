#!/usr/bin/env python3

# Copyright (c) 2026
# All rights reserved.

import argparse
import json
import sys
from pathlib import Path

import yaml

try:
    from .model import (
        compare_results,
        evaluate_config,
        load_config,
        load_records,
        load_sram_records,
    )
except ImportError:
    from model import (
        compare_results,
        evaluate_config,
        load_config,
        load_records,
        load_sram_records,
    )


DEFAULT_DATASHEET = (
    Path(__file__).resolve().parent / "data" / ("ppa_augmented_v1.csv")
)
DEFAULT_SRAM_DATASHEET = (
    Path(__file__).resolve().parent / "data" / "sram_v1.csv"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--datasheet", type=Path, default=DEFAULT_DATASHEET)
    parser.add_argument(
        "--sram-datasheet", type=Path, default=DEFAULT_SRAM_DATASHEET
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.configs) not in (1, 2):
        parser.error(
            "provide one config to evaluate or two configs to compare"
        )
    return args


def _module_text(module):
    record = module["selected_record"]
    fallback = module["selection"]["timing_fallback"]
    return (
        f"  {module['kind']}: instances={module['instance_count']} "
        f"record={record['record_id']} power={module['power_w']:.6f} W "
        f"area={module['area_mm2']:.6f} mm^2 "
        f"timing_met={record['timing_met']} "
        f"timing_fallback={fallback}"
    )


def _sram_text(module):
    record = module["selected_record"]
    return (
        f"  sram: instances={module['instance_count']} "
        f"requested={module['requested_capacity_bytes'] / 1024:.3f} KiB "
        f"provisioned="
        f"{module['provisioned_capacity_bytes'] / 1024:.3f} KiB "
        f"record={record['record_id']} "
        f"static_power={module['static_power_mw']:.6f} mW "
        f"area={module['area_mm2']:.6f} mm^2"
    )


def _evaluation_text(result):
    lines = [
        (
            f"{result['name']}: frequency="
            f"{result['system']['frequency_mhz']:.0f} MHz "
            f"pdk={result['system']['pdk']} "
            f"flow={result['system']['flow']}"
        ),
        _module_text(result["modules"]["vector"]),
        _module_text(result["modules"]["tensor"]),
        _sram_text(result["modules"]["sram"]),
        (
            f"  total: power={result['totals']['power_w']:.6f} W "
            f"area={result['totals']['area_mm2']:.6f} mm^2 "
            "sram_power=static-only"
        ),
    ]
    lines.extend(f"  WARNING: {warning}" for warning in result["warnings"])
    return "\n".join(lines)


def render_text(result):
    if "comparison" not in result:
        return _evaluation_text(result)

    comparison = result["comparison"]
    baseline = result["baseline"]
    candidate = result["candidate"]
    baseline_totals = baseline["totals"]
    candidate_totals = candidate["totals"]
    separator = "=" * 78
    summary = [
        separator,
        (
            f"PPA COMPARISON: {baseline['name']} (baseline) -> "
            f"{candidate['name']} (candidate)"
        ),
        (
            f"Architecture: {baseline['system']['frequency_mhz']:.0f} MHz, "
            "32 x ara_sys + 16 x Mesh_BOTH_32x32 + 256 KiB SRAM"
        ),
        separator,
        (
            f"{'Metric':<12}{baseline['name']:>16}{candidate['name']:>16}"
            f"{'Ratio':>14}{'Improvement':>18}"
        ),
        "-" * 78,
        (
            f"{'Power (W)':<12}{baseline_totals['power_w']:>16.6f}"
            f"{candidate_totals['power_w']:>16.6f}"
            f"{comparison['power_ratio']:>14.6f}"
            f"{comparison['power_improvement_pct']:>17.3f}%"
        ),
        (
            f"{'Area (mm^2)':<12}{baseline_totals['area_mm2']:>16.6f}"
            f"{candidate_totals['area_mm2']:>16.6f}"
            f"{comparison['area_ratio']:>14.6f}"
            f"{comparison['area_improvement_pct']:>17.3f}%"
        ),
        separator,
        "Positive improvement means that F7 uses less Power or Area.",
        "",
        "Module breakdown and timing status:",
    ]
    return "\n".join(
        (
            *summary,
            _evaluation_text(result["baseline"]),
            _evaluation_text(result["candidate"]),
        )
    )


def main():
    args = parse_args()
    try:
        records = load_records(args.datasheet)
        sram_records = load_sram_records(args.sram_datasheet)
        results = [
            evaluate_config(
                load_config(config_path),
                records,
                args.datasheet,
                sram_records,
                args.sram_datasheet,
            )
            for config_path in args.configs
        ]
        result = (
            results[0]
            if len(results) == 1
            else compare_results(results[0], results[1])
        )
    except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
        print(f"ppa_eval.py: error: {error}", file=sys.stderr)
        return 1

    rendered = (
        json.dumps(result, indent=2) + "\n"
        if args.format == "json"
        else render_text(result) + "\n"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
