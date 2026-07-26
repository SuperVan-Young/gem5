#!/usr/bin/env python3

# Copyright (c) 2026
# All rights reserved.

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = Path(__file__).resolve().parent
TESTCASE_ROOT = (
    ROOT
    / "tests/gem5/npu/testcases/FlashAttentionV2"
    / ("operator_perf_q256_kv1024_d128")
)
CONFIG = TESTCASE_ROOT / "config.py"
WORKLOAD = (
    TESTCASE_ROOT
    / "bin"
    / ("flash_attention_v2_operator_perf_q256_kv1024_d128_riscv")
)
PROFILE_PARSER = ROOT / "tests/gem5/npu/tools/parse_npu_profile.py"
PROFILE_RENDERER = ROOT / "tests/gem5/npu/tools/render_npu_profile.py"
SUMMARIZER = TESTCASE_ROOT / "summarize_perf.py"
SCENARIO = "flash_attention_v2_operator_perf_q256_kv1024_d128"


class ConfigError(ValueError):
    pass


def _mapping(value, path):
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a mapping")
    return value


def _fields(value, allowed, path):
    unknown = set(value) - set(allowed)
    if unknown:
        raise ConfigError(f"{path} has unknown fields: {sorted(unknown)}")


def _positive_int(value, path):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{path} must be a positive integer")
    return value


def _positive_number(value, path):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a positive number")
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{path} must be a positive number")
    return value


def _string(value, path):
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{path} must be a non-empty string")
    return value


def _load_yaml(path):
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(value, "root")


def load_hardware(path):
    config = _load_yaml(path)
    _fields(
        config,
        {
            "schema_version",
            "name",
            "system",
            "compute",
            "memory",
            "simulation",
        },
        "root",
    )
    if config.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")
    system = _mapping(config.get("system"), "system")
    compute = _mapping(config.get("compute"), "compute")
    vector = _mapping(compute.get("vector"), "compute.vector")
    tensor = _mapping(compute.get("tensor"), "compute.tensor")
    memory = _mapping(config.get("memory"), "memory")
    _fields(memory, {"sram"}, "memory")
    sram = _mapping(memory.get("sram"), "memory.sram")
    _fields(
        sram,
        {"capacity_bytes", "port_groups", "simultaneous_read_write"},
        "memory.sram",
    )
    ppa_sram_capacity = _positive_int(
        sram.get("capacity_bytes"), "memory.sram.capacity_bytes"
    )
    simulation = _mapping(config.get("simulation"), "simulation")
    if simulation.get("schema_version") != 1:
        raise ConfigError("simulation.schema_version must be 1")
    simulation_clock_mhz = _positive_number(
        simulation.get("clock_mhz"), "simulation.clock_mhz"
    )
    ppa_frequency_mhz = _positive_number(
        system.get("frequency_mhz"), "system.frequency_mhz"
    )
    if simulation_clock_mhz != ppa_frequency_mhz:
        raise ConfigError(
            "simulation.clock_mhz must match system.frequency_mhz"
        )

    required_sections = {
        "memory",
        "spm",
        "interconnect",
        "mpu",
        "vpu",
        "lut",
    }
    _fields(
        simulation,
        {"schema_version", "clock_mhz", *required_sections},
        "simulation",
    )
    missing = sorted(required_sections - set(simulation))
    if missing:
        raise ConfigError(f"simulation is missing sections: {missing}")
    for section in required_sections:
        _mapping(simulation[section], f"simulation.{section}")

    array_dim = _positive_int(
        simulation["mpu"].get("array_dim"), "simulation.mpu.array_dim"
    )
    ppa_array_dim = _positive_number(
        tensor.get("array_dim"), "compute.tensor.array_dim"
    )
    if array_dim != ppa_array_dim:
        raise ConfigError(
            "simulation.mpu.array_dim must match compute.tensor.array_dim"
        )
    dlen_bytes = _positive_int(
        simulation["vpu"].get("dlen_bytes"),
        "simulation.vpu.dlen_bytes",
    )
    vector_elements = _positive_number(
        vector.get("fp32_elements_per_cycle"),
        "compute.vector.fp32_elements_per_cycle",
    )
    if dlen_bytes % 4 != 0 or dlen_bytes // 4 != vector_elements:
        raise ConfigError(
            "simulation.vpu.dlen_bytes / 4 must match "
            "compute.vector.fp32_elements_per_cycle"
        )
    spm = simulation["spm"]
    _positive_int(spm.get("base_address"), "simulation.spm.base_address")
    _positive_int(spm.get("size_bytes"), "simulation.spm.size_bytes")

    return {
        "name": _string(config.get("name"), "name"),
        "ppa_frequency_mhz": ppa_frequency_mhz,
        "ppa_sram_capacity_bytes": ppa_sram_capacity,
        "array_dim": array_dim,
        "tensor_ops_per_cycle": 2 * array_dim * array_dim,
        "vector_fp32_elements_per_cycle": vector_elements,
        "vector_fp32_flops_per_cycle": 2 * vector_elements,
        "simulation": simulation,
    }


def load_task(path):
    config = _load_yaml(path)
    _fields(
        config,
        {"schema_version", "name", "operator", "attention", "tiling"},
        "root",
    )
    if config.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")
    if config.get("operator") != "flash_attention_v2":
        raise ConfigError("operator must be flash_attention_v2")
    attention = _mapping(config.get("attention"), "attention")
    _fields(
        attention,
        {
            "q",
            "kv",
            "d",
            "batch",
            "heads",
            "input_dtype",
            "accumulation_dtype",
            "softmax_dtype",
        },
        "attention",
    )
    expected = {
        "batch": 1,
        "heads": 1,
        "input_dtype": "int8",
        "accumulation_dtype": "int32",
        "softmax_dtype": "float32",
    }
    for field, value in expected.items():
        if attention.get(field) != value:
            raise ConfigError(f"attention.{field} must be {value!r}")
    shape = {
        field: _positive_int(attention.get(field), f"attention.{field}")
        for field in ("q", "kv", "d")
    }
    if shape["d"] > 255:
        raise ConfigError("attention.d must be <= 255")
    if shape["q"] * shape["kv"] > 16_000_000:
        raise ConfigError("attention.q * attention.kv is too large")
    tiling = _mapping(config.get("tiling"), "tiling")
    _fields(tiling, {"q", "kv"}, "tiling")
    br = _positive_int(tiling.get("q"), "tiling.q")
    bc = _positive_int(tiling.get("kv"), "tiling.kv")
    if br != 128 or bc != 128:
        raise ConfigError("tiling.q and tiling.kv must both be 128 for V1")
    return {
        "name": _string(config.get("name"), "name"),
        **shape,
        "br": br,
        "bc": bc,
        **expected,
    }


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(argv, *, stdout=None, stderr=None):
    return subprocess.run(
        [str(value) for value in argv],
        cwd=ROOT,
        text=True,
        stdout=stdout,
        stderr=stderr,
        check=False,
    )


def _default_output(hardware, task):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        SIM_ROOT / "results" / f"{hardware['name']}__{task['name']}__{stamp}"
    )


def _display_path(path):
    host_root = os.environ.get("GEM5_HOST_ROOT")
    if host_root:
        try:
            return str(Path(host_root) / path.resolve().relative_to(ROOT))
        except ValueError:
            pass
    return str(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", required=True, type=Path)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--gem5-binary",
        type=Path,
        default=ROOT / "build/RISCV/gem5.opt",
    )
    parser.add_argument("--rebuild-workload", action="store_true")
    parser.add_argument("--expect-utilization-min", type=float)
    parser.add_argument("--expect-utilization-max", type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        hardware_path = args.hardware.resolve(strict=True)
        task_path = args.task.resolve(strict=True)
        hardware = load_hardware(hardware_path)
        task = load_task(task_path)
    except (OSError, ConfigError, yaml.YAMLError) as error:
        print(f"gem5-fa-sim: input error: {error}", file=sys.stderr)
        return 2

    gem5_binary = args.gem5_binary.resolve()
    if not gem5_binary.is_file():
        print(
            "gem5-fa-sim: missing gem5 binary; build with "
            "scons -j32 build/RISCV/gem5.opt",
            file=sys.stderr,
        )
        return 2
    if args.rebuild_workload or not WORKLOAD.is_file():
        result = _run(["make", "-C", TESTCASE_ROOT, "-j32"])
        if result.returncode != 0:
            return result.returncode

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _default_output(hardware, task)
    )
    if output_dir.exists():
        print(
            f"gem5-fa-sim: output directory already exists: {output_dir}",
            file=sys.stderr,
        )
        return 2
    output_dir.mkdir(parents=True)
    gem5_output = output_dir / "gem5"
    gem5_output.mkdir()
    hardware_snapshot = output_dir / "hardware.yaml"
    task_snapshot = output_dir / "task.yaml"
    resolved_config = output_dir / "resolved_simulation.json"
    profile_log = output_dir / "npu_profile.log"
    profile_json = output_dir / "npu_profile.json"
    profile_html = output_dir / "npu_profile.html"
    simout = output_dir / "simout.txt"
    simerr = output_dir / "simerr.txt"
    summary_path = output_dir / "performance_summary.json"
    manifest_path = output_dir / "manifest.json"
    shutil.copyfile(hardware_path, hardware_snapshot)
    shutil.copyfile(task_path, task_snapshot)
    resolved_config.write_text(
        json.dumps(hardware["simulation"], indent=2) + "\n",
        encoding="utf-8",
    )

    command = [
        gem5_binary,
        "-d",
        gem5_output,
        "--debug-flags=NPUProfile",
        f"--debug-file={profile_log}",
        CONFIG,
        "--binary",
        WORKLOAD,
        "--scenario",
        SCENARIO,
        "--q",
        task["q"],
        "--kv",
        task["kv"],
        "--d",
        task["d"],
        "--br",
        task["br"],
        "--bc",
        task["bc"],
        "--profile-log",
        profile_log,
        "--simulation-config",
        resolved_config,
    ]
    manifest = {
        "schema_version": 1,
        "status": "RUNNING",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "hardware_sha256": _sha256(hardware_path),
        "task_sha256": _sha256(task_path),
        "workload_sha256": _sha256(WORKLOAD),
        "command": [str(value) for value in command],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    with simout.open("w", encoding="utf-8") as out, simerr.open(
        "w", encoding="utf-8"
    ) as err:
        result = _run(command, stdout=out, stderr=err)
    manifest["gem5_exit_code"] = result.returncode
    if result.returncode != 0:
        manifest["status"] = "FAIL"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"gem5-fa-sim: gem5 failed; see {simerr}", file=sys.stderr)
        return result.returncode or 1

    tools = [
        [
            sys.executable,
            PROFILE_PARSER,
            "--input",
            profile_log,
            "--output",
            profile_json,
        ],
        [
            sys.executable,
            PROFILE_RENDERER,
            "--input",
            profile_json,
            "--output",
            profile_html,
        ],
        [
            sys.executable,
            SUMMARIZER,
            "--simout",
            simout,
            "--profile-log",
            profile_log,
            "--profile-json",
            profile_json,
            "--profile-html",
            profile_html,
            "--output",
            summary_path,
            "--clock-mhz",
            hardware["simulation"]["clock_mhz"],
            "--array-dim",
            hardware["array_dim"],
        ],
    ]
    for tool in tools:
        tool_result = _run(
            tool, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        if tool_result.returncode != 0:
            manifest["status"] = "FAIL"
            manifest["postprocess_error"] = tool_result.stderr
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            print(tool_result.stderr, file=sys.stderr)
            return tool_result.returncode

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = "PASS"
    summary["hardware"] = {
        "name": hardware["name"],
        "ppa_frequency_mhz": hardware["ppa_frequency_mhz"],
        "ppa_sram_capacity_bytes": hardware["ppa_sram_capacity_bytes"],
        "simulation_spm_size_bytes": hardware["simulation"]["spm"][
            "size_bytes"
        ],
        "simulation_clock_mhz": hardware["simulation"]["clock_mhz"],
        "tensor_array_dim": hardware["array_dim"],
        "tensor_ops_per_cycle": hardware["tensor_ops_per_cycle"],
        "vector_fp32_elements_per_cycle": hardware[
            "vector_fp32_elements_per_cycle"
        ],
        "vector_fp32_flops_per_cycle": hardware[
            "vector_fp32_flops_per_cycle"
        ],
    }
    summary["task"] = task
    summary["artifacts"].update(
        {
            "simout": _display_path(simout),
            "profile_log": _display_path(profile_log),
            "profile_json": _display_path(profile_json),
            "profile_html": _display_path(profile_html),
            "manifest": _display_path(manifest_path),
            "resolved_simulation": _display_path(resolved_config),
            "simerr": _display_path(simerr),
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    utilization = summary["performance"]["utilization_percent"]
    acceptance_error = None
    if (
        args.expect_utilization_min is not None
        and utilization < args.expect_utilization_min
    ):
        acceptance_error = (
            f"utilization {utilization:.3f}% is below "
            f"{args.expect_utilization_min:.3f}%"
        )
    if (
        args.expect_utilization_max is not None
        and utilization > args.expect_utilization_max
    ):
        acceptance_error = (
            f"utilization {utilization:.3f}% is above "
            f"{args.expect_utilization_max:.3f}%"
        )

    manifest["status"] = "FAIL" if acceptance_error else "PASS"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["performance_summary"] = str(summary_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    profile = summary["profile"]
    performance = summary["performance"]
    print("FlashAttentionV2 cycle simulation: " + manifest["status"])
    print(
        f"Hardware: {hardware['name']} "
        f"(PPA {hardware['ppa_frequency_mhz']:.0f} MHz, "
        f"simulation {hardware['simulation']['clock_mhz']:.0f} MHz)"
    )
    print(
        f"Task: q={task['q']} kv={task['kv']} d={task['d']} "
        f"BR={task['br']} BC={task['bc']} batch=1 heads=1"
    )
    print("")
    print(f"{'Metric':<24}{'Cycles':>14}")
    print("-" * 38)
    print(
        f"{'Operator busy':<24}" f"{performance['operator_busy_cycles']:>14,d}"
    )
    for label, key in (
        ("QK busy", "qk_fast_matmul"),
        ("Softmax busy", "fast_online_softmax"),
        ("PV busy", "pv_fast_matmul"),
    ):
        print(f"{label:<24}{profile[key]['busy_cycles']:>14,d}")
    print("")
    print(
        f"Tensor throughput: {performance['effective_tops']:.6f} TOPS "
        f"({utilization:.3f}% of "
        f"{performance['theoretical_tops']:.3f} TOPS peak)"
    )
    print(f"Profile HTML: {_display_path(profile_html)}")
    print(f"Summary JSON: {_display_path(summary_path)}")
    print(f"RESULT_DIR={_display_path(output_dir)}")
    if acceptance_error:
        print(f"gem5-fa-sim: {acceptance_error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
