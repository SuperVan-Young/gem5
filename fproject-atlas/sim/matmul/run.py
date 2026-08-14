#!/usr/bin/env python3

# Copyright (c) 2026
# All rights reserved.

import argparse
import json
import math
import subprocess
import sys
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
MATMUL_ROOT = Path(__file__).resolve().parent
HW_ROOT = REPO_ROOT / "fproject-atlas/hw"
PROFILE_PARSER = REPO_ROOT / "tests/gem5/npu/tools/parse_npu_profile.py"
PROFILE_RENDERER = REPO_ROOT / "tests/gem5/npu/tools/render_npu_profile.py"
sys.path.insert(0, str(HW_ROOT))


def load_hardware(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("hardware config must use schema_version 1")
    return config


def compare_cycles(m, n, k, array_dim, gem5_busy_cycles, gem5_macs):
    mac_count = m * n * k
    naive_cycles = math.ceil(mac_count / (array_dim * array_dim))
    tile_count = math.ceil(m / array_dim) * math.ceil(n / array_dim)
    compute_cycles = tile_count * k
    return {
        "mac_count": mac_count,
        "mac_capacity_per_cycle": array_dim * array_dim,
        "naive_cycles": naive_cycles,
        "tile_count": tile_count,
        "gem5_compute_cycles": compute_cycles,
        "gem5_busy_cycles": gem5_busy_cycles,
        "gem5_mac_count": gem5_macs,
        "compute_over_naive": compute_cycles / naive_cycles,
        "busy_over_naive": gem5_busy_cycles / naive_cycles,
        "busy_over_compute": gem5_busy_cycles / compute_cycles,
    }


def summarize_profile(profile, clock_mhz):
    macro_events = [
        event
        for event in profile["events"]
        if not event["axis"].endswith(".uops")
    ]
    uop_events = [
        event for event in profile["events"] if event["axis"].endswith(".uops")
    ]
    start_tick = min(event["start_tick"] for event in macro_events)
    end_tick = max(event["end_tick"] for event in macro_events)
    ticks_per_cycle = 1_000_000 / clock_mhz
    return {
        "axis_count": profile["axis_count"],
        "event_count": profile["event_count"],
        "macro_event_count": len(macro_events),
        "uop_event_count": len(uop_events),
        "span_ticks": end_tick - start_tick,
        "span_cycles": int((end_tick - start_tick) / ticks_per_cycle),
    }


def validate_shape(m, n, k):
    if min(m, n, k) <= 0:
        raise ValueError("M, N, and K must be positive")
    if k > 255 or k % 16 != 0:
        raise ValueError("K must be a multiple of 16 and at most 255")


def run_command(argv, **kwargs):
    return subprocess.run(
        [str(value) for value in argv],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        **kwargs,
    )


def host_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", required=True, type=Path)
    parser.add_argument("--m", required=True, type=int)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--gem5-binary",
        type=Path,
        default=REPO_ROOT / "build/RISCV/gem5.opt",
    )
    parser.add_argument("--rebuild-workload", action="store_true")
    args = parser.parse_args()

    try:
        hardware_path = args.hardware.resolve(strict=True)
        hardware = load_hardware(hardware_path)
        validate_shape(args.m, args.n, args.k)
    except (OSError, ValueError) as error:
        print(f"atlas-matmul: input error: {error}", file=sys.stderr)
        return 2

    gem5_binary = args.gem5_binary.resolve()
    if not gem5_binary.is_file():
        print(
            "atlas-matmul: missing gem5 binary; build with "
            "scons -j32 build/RISCV/gem5.opt",
            file=sys.stderr,
        )
        return 2

    workload = MATMUL_ROOT / "bin/atlas_matmul_riscv"
    if args.rebuild_workload or not workload.is_file():
        built = run_command(["make", "-C", MATMUL_ROOT, "-j32"])
        if built.returncode != 0:
            return built.returncode

    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = (
            REPO_ROOT
            / "fproject-atlas/results/matmul"
            / f"{hardware['name']}__m{args.m}_n{args.n}_k{args.k}__{stamp}"
        )
    else:
        output_dir = args.output_dir.resolve()
    if output_dir.exists():
        print(
            f"atlas-matmul: output already exists: {output_dir}",
            file=sys.stderr,
        )
        return 2
    output_dir.mkdir(parents=True)

    simout = output_dir / "simout.txt"
    simerr = output_dir / "simerr.txt"
    result_path = output_dir / "result.json"
    profile_log = output_dir / "npu_profile.log"
    profile_json = output_dir / "npu_profile.json"
    profile_html = output_dir / "npu_profile.html"
    command = [
        gem5_binary,
        "-d",
        output_dir / "gem5",
        "--debug-flags=NPUProfile",
        f"--debug-file={profile_log}",
        Path(__file__).resolve(),
        "--gem5-run",
        "--hardware",
        hardware_path,
        "--binary",
        workload,
        "--m",
        args.m,
        "--n",
        args.n,
        "--k",
        args.k,
        "--output",
        result_path,
    ]
    with simout.open("w", encoding="utf-8") as stdout, simerr.open(
        "w", encoding="utf-8"
    ) as stderr:
        simulated = run_command(command, stdout=stdout, stderr=stderr)
    if simulated.returncode != 0 or not result_path.is_file():
        print(f"atlas-matmul: gem5 failed; see {simerr}", file=sys.stderr)
        return simulated.returncode or 1

    for tool in (
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
    ):
        processed = run_command(tool)
        if processed.returncode != 0:
            return processed.returncode

    result = json.loads(result_path.read_text(encoding="utf-8"))
    profile = json.loads(profile_json.read_text(encoding="utf-8"))
    result["profiling"] = {
        **summarize_profile(profile, hardware["clock_mhz"]),
        "raw_log": str(profile_log),
        "json": str(profile_json),
        "html": str(profile_html),
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    comparison = result["comparison"]
    print("Atlas Matmul simulation: PASS")
    print(
        f"shape={args.m}x{args.n}x{args.k} "
        f"array={result['array_dim']}x{result['array_dim']}"
    )
    print(
        f"naive={comparison['naive_cycles']:,} cycles, "
        f"gem5-compute={comparison['gem5_compute_cycles']:,} cycles "
        f"({comparison['compute_over_naive']:.3f}x), "
        f"gem5-busy={comparison['gem5_busy_cycles']:,} cycles "
        f"({comparison['busy_over_naive']:.3f}x)"
    )
    print(
        f"profile-events={profile['event_count']}, "
        f"profile-span={result['profiling']['span_cycles']:,} cycles, "
        f"profile-html={profile_html}"
    )
    print(f"RESULT_JSON={result_path}")
    return 0


def gem5_main():
    from top import (
        build_top,
        map_workload,
    )

    import m5

    parser = argparse.ArgumentParser()
    parser.add_argument("--gem5-run", action="store_true")
    parser.add_argument("--hardware", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--m", required=True, type=int)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    hardware = load_hardware(args.hardware)
    validate_shape(args.m, args.n, args.k)
    array_dim = hardware["mpu"]["array_dim"]
    top = build_top(
        hardware,
        args.binary,
        [
            args.m,
            args.n,
            args.k,
            array_dim,
            hardware["memory"]["spm"]["base_address"],
            hardware["memory"]["spm"]["size_bytes"],
        ],
    )
    m5.instantiate()
    map_workload(top, hardware)
    exit_event = m5.simulate()
    if exit_event.getCode() != 0:
        raise RuntimeError(
            f"workload failed: {exit_event.getCause()} "
            f"(code {exit_event.getCode()})"
        )

    mpu = top.system.mpu
    comparison = compare_cycles(
        args.m,
        args.n,
        args.k,
        array_dim,
        int(mpu.busyCycles()),
        int(mpu.totalMacOps()),
    )
    if comparison["gem5_mac_count"] != comparison["mac_count"]:
        raise RuntimeError("gem5 MAC count does not match M*N*K")

    result = {
        "schema_version": 1,
        "operator": "matmul",
        "hardware": hardware["name"],
        "clock_mhz": hardware["clock_mhz"],
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "array_dim": array_dim,
        "comparison": comparison,
        "exit_cause": exit_event.getCause(),
        "exit_code": exit_event.getCode(),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("ATLAS_MATMUL_PASS")


if "--gem5-run" in sys.argv:
    gem5_main()
elif __name__ == "__main__":
    sys.exit(host_main())
