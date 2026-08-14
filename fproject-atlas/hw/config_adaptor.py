#!/usr/bin/env python3

# Copyright (c) 2026
# All rights reserved.

"""Convert an ATLAS chip configuration into a gem5 hardware configuration."""

import argparse
import copy
import math
import re
from fractions import Fraction
from pathlib import Path

import yaml

ATLAS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ATLAS_ROOT / "configs/gem5_template.yaml"

# gem5_template.yaml represents this ATLAS design point. Other designs are
# converted by exact relative scaling; an inexact mapping is an error.
REFERENCE_MAC_NUM = 8192
REFERENCE_VEC_NUM = 512
REFERENCE_BUFFER_KIB = 3072
REFERENCE_NOC_FLIT_BYTES = 64

HBDRAM_TIMINGS = {
    "HBDRAM_500Mbps": {
        "rate_mbps": 500,
        "nBL": 1,
        "nCL": 8,
        "nRCDRD": 8,
        "tCK_ps": 2000,
    },
    "HBDRAM_400Mbps": {
        "rate_mbps": 400,
        "nBL": 1,
        "nCL": 7,
        "nRCDRD": 7,
        "tCK_ps": 2500,
    },
}


class HexInt(int):
    """An integer emitted in hexadecimal form by the YAML dumper."""


class ConfigDumper(yaml.SafeDumper):
    pass


ConfigDumper.add_representer(
    HexInt,
    lambda dumper, value: dumper.represent_scalar(
        "tag:yaml.org,2002:int", hex(value)
    ),
)


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def positive_fraction(name, value):
    result = Fraction(str(value))
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return result


def exact_int(name, value):
    if value.denominator != 1:
        raise ValueError(f"{name} cannot be represented exactly: {value}")
    return value.numerator


def exact_scaled_int(name, base, scale):
    return exact_int(name, Fraction(base) * scale)


def format_number(value):
    if value.denominator == 1:
        return str(value.numerator)
    return f"{float(value):g}"


def resolve_reference(chip_path, reference):
    reference = Path(reference)
    if reference.is_absolute() and reference.is_file():
        return reference
    for parent in (chip_path.parent, *chip_path.parents):
        candidate = parent / reference
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"cannot resolve ATLAS reference {reference} from {chip_path}"
    )


def decode_hbdram(chip_path, dram_config):
    reference = resolve_reference(chip_path, dram_config["config_path"])
    ramulator = load_yaml(reference)["MemorySystem"]["DRAM"]
    org = ramulator["org"]
    timing_name = ramulator["timing"]["preset"]

    match = re.fullmatch(
        r"HBDRAM_(\d+)(Gb|Mb)_(\d+)pin(?:_.*)?", org["preset"]
    )
    if match is None:
        raise ValueError(f"unsupported HBDRAM organization: {org['preset']}")
    density = int(match.group(1))
    density_mibits = density * 1024 if match.group(2) == "Gb" else density
    dq_bits = int(match.group(3))
    channels = int(org["channel"])

    if timing_name not in HBDRAM_TIMINGS:
        raise ValueError(f"unsupported HBDRAM timing preset: {timing_name}")
    timing = HBDRAM_TIMINGS[timing_name]

    capacity_bytes = density_mibits * 1024 * 1024 // 8 * channels
    bandwidth_gbs = Fraction(
        timing["rate_mbps"] * dq_bits * channels,
        8 * 1000,
    )
    latency_ns = Fraction(
        (timing["nRCDRD"] + timing["nCL"] + timing["nBL"]) * timing["tCK_ps"],
        1000,
    )
    return capacity_bytes, bandwidth_gbs, latency_ns


def adapt_mpu(output, template, matrix):
    mac_scale = (
        positive_fraction("mac_num", matrix["mac_num"]) / REFERENCE_MAC_NUM
    )
    scaled_array_area = exact_scaled_int(
        "MPU array area",
        template["mpu"]["array_dim"] ** 2,
        mac_scale,
    )
    array_dim = math.isqrt(scaled_array_area)
    if array_dim * array_dim != scaled_array_area:
        raise ValueError(
            f"mac_num={matrix['mac_num']} maps to non-square MPU area "
            f"{scaled_array_area}"
        )
    output["mpu"]["array_dim"] = array_dim
    for key in ("a_buffer_bytes", "b_buffer_bytes", "c_buffer_bytes"):
        output["mpu"][key] = exact_scaled_int(
            f"mpu.{key}", template["mpu"][key], mac_scale
        )


def adapt_vector(output, template, vector):
    vec_scale = (
        positive_fraction("vec_num", vector["vec_num"]) / REFERENCE_VEC_NUM
    )
    dlen_bytes = exact_scaled_int(
        "vector.dlen_bytes", template["vector"]["dlen_bytes"], vec_scale
    )
    if dlen_bytes % 4 != 0:
        raise ValueError(
            f"dlen_bytes must be divisible by 4, got {dlen_bytes}"
        )
    output["vector"]["dlen_bytes"] = dlen_bytes
    output["vector"]["mem_side_ports"] = exact_scaled_int(
        "vector.mem_side_ports",
        template["vector"]["mem_side_ports"],
        vec_scale,
    )
    return dlen_bytes


def adapt_spm(output, template, architecture, buffer, dlen_bytes):
    buffer_scale = (
        positive_fraction("buffer_size", buffer["buffer_size"])
        / REFERENCE_BUFFER_KIB
    )
    output["memory"]["spm"]["size_bytes"] = exact_int(
        "memory.spm.size_bytes",
        positive_fraction("buffer_size", buffer["buffer_size"]) * 1024,
    )
    combined_bytes_per_cycle = positive_fraction(
        "buffer.read_bw", buffer["read_bw"]
    ) + positive_fraction("buffer.write_bw", buffer["write_bw"])
    pipeline_ports = exact_int(
        "memory.spm.pipeline_ports", combined_bytes_per_cycle / dlen_bytes
    )
    output["memory"]["spm"]["pipeline_ports"] = pipeline_ports
    per_port_bytes_per_cycle = combined_bytes_per_cycle / pipeline_ports
    spm_bandwidth_gbs = (
        per_port_bytes_per_cycle
        * positive_fraction("frequency", architecture["frequency"])
        / 1000
    )
    output["memory"]["spm"][
        "bandwidth"
    ] = f"{format_number(spm_bandwidth_gbs)}GB/s"

    local_buffer_stride = exact_scaled_int(
        "vector.local_buffer_stride",
        template["vector"]["local_buffer_stride"],
        buffer_scale,
    )
    if local_buffer_stride % dlen_bytes != 0:
        raise ValueError(
            "scaled local_buffer_stride is not divisible by dlen_bytes: "
            f"{local_buffer_stride} vs {dlen_bytes}"
        )
    output["vector"]["local_buffer_stride"] = local_buffer_stride


def adapt_interconnect(output, template, architecture):
    if "noc" in architecture:
        flit_scale = (
            positive_fraction(
                "noc.flit_size", architecture["noc"]["flit_size"]
            )
            / REFERENCE_NOC_FLIT_BYTES
        )
        output["interconnect"]["width_bytes"] = exact_scaled_int(
            "interconnect.width_bytes",
            template["interconnect"]["width_bytes"],
            flit_scale,
        )


def adapt_dram(output, architecture, chip_path):
    total_capacity_bytes, total_bandwidth_gbs, latency_ns = decode_hbdram(
        chip_path, architecture["dram"]
    )
    core_num = positive_fraction("core_num", architecture["core_num"])
    per_core_capacity_bytes = exact_int(
        "per-core DRAM capacity", Fraction(total_capacity_bytes) / core_num
    )
    per_core_bandwidth_gbs = total_bandwidth_gbs / core_num

    output["memory"]["dram"]["mapped_size_bytes"] = per_core_capacity_bytes
    output["memory"]["dram"]["capacity_bytes"] = per_core_capacity_bytes
    output["memory"]["dram"][
        "bandwidth"
    ] = f"{format_number(per_core_bandwidth_gbs)}GB/s"
    output["memory"]["dram"]["latency"] = f"{format_number(latency_ns)}ns"


def adapt(atlas_config, template, chip_path):
    architecture = atlas_config["architecture"]
    core = architecture["core"]
    output = copy.deepcopy(template)
    output["name"] = f"{chip_path.stem}-gem5"
    output["clock_mhz"] = architecture["frequency"]

    adapt_mpu(output, template, core["matrix"])
    dlen_bytes = adapt_vector(output, template, core["vector"])
    adapt_spm(output, template, architecture, core["buffer"], dlen_bytes)
    adapt_interconnect(output, template, architecture)
    adapt_dram(output, architecture, chip_path)

    output["memory"]["spm"]["base_address"] = HexInt(
        output["memory"]["spm"]["base_address"]
    )
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Convert an ATLAS attention_chip.yaml to gem5 YAML."
    )
    parser.add_argument("atlas_chip", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    args = parser.parse_args()

    atlas_config = load_yaml(args.atlas_chip)
    template = load_yaml(args.template)
    output = adapt(atlas_config, template, args.atlas_chip)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.dump(
            output,
            Dumper=ConfigDumper,
            sort_keys=False,
            default_flow_style=False,
            indent=2,
            explicit_start=True,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
