#!/usr/bin/env python3

# Copyright (c) 2026
# All rights reserved.

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SOFTMAX_ROOT = Path(__file__).resolve().parent
HW_ROOT = REPO_ROOT / "fproject-atlas/hw"
PROFILE_PARSER = REPO_ROOT / "tests/gem5/npu/tools/parse_npu_profile.py"
PROFILE_RENDERER = REPO_ROOT / "tests/gem5/npu/tools/render_npu_profile.py"
sys.path.insert(0, str(HW_ROOT))


def load_hardware(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("hardware config must use schema_version 1")
    return config


def compare_cycles(rows, cols, vector, gem5_busy_cycles):
    atlas_ops_per_element = 9
    atlas_vec_count = atlas_ops_per_element * rows * cols
    capacity = (vector["dlen_bytes"] // 4) / vector["float32_cycles_per_dlen"]
    naive_cycles = math.ceil(atlas_vec_count / capacity)
    return {
        "model": "ATLAS edge softmax",
        "atlas_ops_per_element": atlas_ops_per_element,
        "atlas_vec_count": atlas_vec_count,
        "atlas_rows": rows,
        "atlas_context_length": cols,
        "vector_vec_num": capacity,
        "naive_cycles": naive_cycles,
        "gem5_busy_cycles": gem5_busy_cycles,
        "gem5_over_naive": gem5_busy_cycles / naive_cycles,
    }


def summarize_profile(profile, clock_mhz):
    events = [
        event
        for event in profile["events"]
        if not event["axis"].endswith(".uops")
    ]
    start_tick = min(event["start_tick"] for event in events)
    end_tick = max(event["end_tick"] for event in events)
    ticks_per_cycle = 1_000_000 / clock_mhz
    opcode_counts = Counter(event["begin"]["opcode"] for event in events)
    opcode_ticks = Counter()
    for event in events:
        opcode_ticks[event["begin"]["opcode"]] += event["duration"]
    opcode_names = {
        1: "add",
        2: "sub",
        3: "mul",
        4: "div",
        5: "scale",
        9: "abs",
        10: "reduce_sum",
        11: "reduce_max",
        14: "exp",
    }
    return {
        "axis_count": profile["axis_count"],
        "event_count": profile["event_count"],
        "macro_event_count": len(events),
        "uop_event_count": profile["event_count"] - len(events),
        "span_cycles": int((end_tick - start_tick) / ticks_per_cycle),
        "busy_cycles": int(
            sum(event["duration"] for event in events) / ticks_per_cycle
        ),
        "opcode_counts": {
            name: opcode_counts[opcode]
            for opcode, name in opcode_names.items()
        },
        "opcode_busy_cycles": {
            name: int(opcode_ticks[opcode] / ticks_per_cycle)
            for opcode, name in opcode_names.items()
        },
    }


def validate_shape(rows, cols, dlen_bytes):
    layout_elements = dlen_bytes // 4
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if rows % layout_elements != 0 or cols % layout_elements != 0:
        raise ValueError(
            f"rows and cols must be multiples of {layout_elements}"
        )


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
    parser.add_argument("--rows", required=True, type=int)
    parser.add_argument("--cols", required=True, type=int)
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
        validate_shape(args.rows, args.cols, hardware["vector"]["dlen_bytes"])
    except (OSError, ValueError) as error:
        print(f"atlas-softmax: input error: {error}", file=sys.stderr)
        return 2

    gem5_binary = args.gem5_binary.resolve()
    if not gem5_binary.is_file():
        print(
            "atlas-softmax: missing gem5 binary; build with "
            "scons -j32 build/RISCV/gem5.opt",
            file=sys.stderr,
        )
        return 2

    workload = SOFTMAX_ROOT / "bin/atlas_online_softmax_riscv"
    if args.rebuild_workload or not workload.is_file():
        built = run_command(["make", "-C", SOFTMAX_ROOT, "-j32"])
        if built.returncode != 0:
            return built.returncode

    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = (
            REPO_ROOT
            / "fproject-atlas/results/softmax"
            / f"{hardware['name']}__r{args.rows}_c{args.cols}__{stamp}"
        )
    else:
        output_dir = args.output_dir.resolve()
    if output_dir.exists():
        print(
            f"atlas-softmax: output already exists: {output_dir}",
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
        "--rows",
        args.rows,
        "--cols",
        args.cols,
        "--output",
        result_path,
    ]
    with simout.open("w", encoding="utf-8") as stdout, simerr.open(
        "w", encoding="utf-8"
    ) as stderr:
        simulated = run_command(command, stdout=stdout, stderr=stderr)
    if simulated.returncode != 0 or not result_path.is_file():
        print(f"atlas-softmax: gem5 failed; see {simerr}", file=sys.stderr)
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
    profile_summary = summarize_profile(profile, hardware["clock_mhz"])
    result["comparison"] = compare_cycles(
        args.rows,
        args.cols,
        hardware["vector"],
        profile_summary["busy_cycles"],
    )
    result["profiling"] = {
        **profile_summary,
        "raw_log": str(profile_log),
        "json": str(profile_json),
        "html": str(profile_html),
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    comparison = result["comparison"]
    print("Atlas OnlineSoftmax simulation: PASS")
    print(
        f"shape={args.rows}x{args.cols} "
        f"vector-capacity={comparison['vector_vec_num']:.0f} "
        "FP32 elements/cycle"
    )
    print(
        f"naive={comparison['naive_cycles']:,} cycles, "
        f"gem5-busy={comparison['gem5_busy_cycles']:,} cycles "
        f"({comparison['gem5_over_naive']:.3f}x)"
    )
    print(
        f"profile-macros={profile_summary['macro_event_count']}, "
        f"profile-span={profile_summary['span_cycles']:,} cycles, "
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
    parser.add_argument("--rows", required=True, type=int)
    parser.add_argument("--cols", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    hardware = load_hardware(args.hardware)
    vector = hardware["vector"]
    validate_shape(args.rows, args.cols, vector["dlen_bytes"])
    top = build_top(
        hardware,
        args.binary,
        [
            args.rows,
            args.cols,
            vector["dlen_bytes"],
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

    vpu = top.system.vpu0
    result = {
        "schema_version": 1,
        "operator": "online_softmax",
        "hardware": hardware["name"],
        "clock_mhz": hardware["clock_mhz"],
        "shape": {"rows": args.rows, "cols": args.cols},
        "vector": {
            "dlen_bytes": vector["dlen_bytes"],
            "float32_cycles_per_dlen": vector["float32_cycles_per_dlen"],
        },
        "vpu": {
            "completed_commands": int(vpu.completedCmdCount()),
            "completed_iterations": int(vpu.completedIterationCount()),
            "max_active_uops": int(vpu.maxActiveMicroOps()),
            "lut_requests": int(vpu.lutRequestCount()),
            "lut_commands": int(vpu.lutCommandCount()),
        },
        "exit_cause": exit_event.getCause(),
        "exit_code": exit_event.getCode(),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("ATLAS_ONLINE_SOFTMAX_PASS")


if "--gem5-run" in sys.argv:
    gem5_main()
elif __name__ == "__main__":
    sys.exit(host_main())
