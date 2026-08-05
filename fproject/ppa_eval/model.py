# Copyright (c) 2026
# All rights reserved.

import csv
import hashlib
import math
from dataclasses import (
    asdict,
    dataclass,
)
from pathlib import Path

import yaml

ARA_LANES = 4
ARA_FP32_ELEMENTS_PER_LANE_PER_CYCLE = 2
OPS_PER_FMA = 2
MESH_DIM = 32
DEFAULT_FLOW = "DC-Innovus"
DEFAULT_UTILIZATION_PCT = 50.0
DEFAULT_ARA_VARIANT = "with_macro"
SRAM_DESIGN = "SRAM_4096x128"


@dataclass(frozen=True)
class PPARecord:
    record_id: str
    source_kind: str
    pdk: str
    flow: str
    design: str
    variant: str
    utilization_pct: float
    target_period_ns: float
    timing_slack_ns: float
    timing_met: bool
    area_um2: float
    area_mm2: float
    power_w: float
    source_row: str
    derived_from: str
    power_scale: float
    area_scale: float
    shared_power_w: float = 0.0
    shared_area_um2: float = 0.0
    shared_area_mm2: float = 0.0
    shared_derived_from: str = ""

    @classmethod
    def from_csv_row(cls, row):
        return cls(
            record_id=row["record_id"],
            source_kind=row["source_kind"],
            pdk=row["pdk"],
            flow=row["flow"],
            design=row["design"],
            variant=row["variant"],
            utilization_pct=float(row["utilization_pct"]),
            target_period_ns=float(row["target_period_ns"]),
            timing_slack_ns=float(row["timing_slack_ns"]),
            timing_met=row["timing_met"].lower() == "true",
            area_um2=float(row["area_um2"]),
            area_mm2=float(row["area_mm2"]),
            power_w=float(row["power_w"]),
            source_row=row["source_row"],
            derived_from=row["derived_from"],
            power_scale=float(row["power_scale"]),
            area_scale=float(row["area_scale"]),
            shared_power_w=float(row.get("shared_power_w") or 0.0),
            shared_area_um2=float(row.get("shared_area_um2") or 0.0),
            shared_area_mm2=float(row.get("shared_area_mm2") or 0.0),
            shared_derived_from=row.get("shared_derived_from", ""),
        )


@dataclass(frozen=True)
class SRAMRecord:
    record_id: str
    source_kind: str
    pdk: str
    design: str
    depth: int
    word_bits: int
    capacity_bytes: int
    max_frequency_mhz: float
    power_w: float
    area_um2: float
    area_mm2: float
    derived_from: str
    power_scale: float
    area_scale: float
    provenance: str

    @classmethod
    def from_csv_row(cls, row):
        try:
            record = cls(
                record_id=row["record_id"],
                source_kind=row["source_kind"],
                pdk=row["pdk"],
                design=row["design"],
                depth=int(row["depth"]),
                word_bits=int(row["word_bits"]),
                capacity_bytes=int(row["capacity_bytes"]),
                max_frequency_mhz=float(row["max_frequency_mhz"]),
                power_w=float(row["power_w"]),
                area_um2=float(row["area_um2"]),
                area_mm2=float(row["area_mm2"]),
                derived_from=row["derived_from"],
                power_scale=float(row["power_scale"]),
                area_scale=float(row["area_scale"]),
                provenance=row["provenance"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid SRAM datasheet row: {error}") from error
        record.validate()
        return record

    def validate(self):
        label = self.record_id or "<missing record_id>"
        if not self.record_id or not self.pdk or not self.provenance:
            raise ValueError(
                f"SRAM record {label} has missing identity fields"
            )
        if self.source_kind not in {"spec", "derived"}:
            raise ValueError(f"SRAM record {label} has invalid source_kind")
        if self.design != SRAM_DESIGN:
            raise ValueError(f"SRAM record {label} has unsupported design")
        if self.depth <= 0 or self.word_bits <= 0 or self.word_bits % 8 != 0:
            raise ValueError(f"SRAM record {label} has invalid organization")
        expected_capacity = self.depth * self.word_bits // 8
        if self.capacity_bytes != expected_capacity:
            raise ValueError(
                f"SRAM record {label} capacity does not match organization"
            )
        positive_values = {
            "max_frequency_mhz": self.max_frequency_mhz,
            "power_w": self.power_w,
            "area_um2": self.area_um2,
            "area_mm2": self.area_mm2,
            "power_scale": self.power_scale,
            "area_scale": self.area_scale,
        }
        for field, value in positive_values.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"SRAM record {label} has invalid {field}")
        if not _close(self.area_mm2, self.area_um2 / 1_000_000.0):
            raise ValueError(
                f"SRAM record {label} has inconsistent area units"
            )
        if self.source_kind == "spec" and self.derived_from:
            raise ValueError(
                f"SRAM record {label} spec source cannot be derived"
            )
        if self.source_kind == "derived" and not self.derived_from:
            raise ValueError(f"SRAM record {label} is missing derived_from")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(path):
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"Datasheet is empty: {path}")
    return [PPARecord.from_csv_row(row) for row in rows]


def load_sram_records(path):
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"SRAM datasheet is empty: {path}")
    records = [SRAMRecord.from_csv_row(row) for row in rows]
    keys = [(record.pdk, record.design) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("SRAM datasheet has duplicate pdk/design records")
    by_id = {record.record_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("SRAM datasheet has duplicate record_id values")
    for record in records:
        if record.source_kind != "derived":
            continue
        source = by_id.get(record.derived_from)
        if source is None:
            raise ValueError(
                f"SRAM record {record.record_id} has unknown derived_from"
            )
        inherited = (
            record.design == source.design
            and record.depth == source.depth
            and record.word_bits == source.word_bits
            and record.capacity_bytes == source.capacity_bytes
            and _close(record.max_frequency_mhz, source.max_frequency_mhz)
        )
        scaled = _close(
            record.power_w,
            source.power_w * record.power_scale,
        ) and _close(record.area_um2, source.area_um2 * record.area_scale)
        if not inherited or not scaled:
            raise ValueError(
                f"SRAM record {record.record_id} derivation is inconsistent"
            )
    return records


def _mapping(value, path):
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def _check_fields(mapping, allowed, path):
    unknown = set(mapping) - set(allowed)
    if unknown:
        raise ValueError(f"{path} has unknown fields: {sorted(unknown)}")


def _positive_number(value, path):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{path} must be finite and greater than zero")
    return float(value)


def _positive_int(value, path):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def load_config(path):
    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "root")
    _check_fields(
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
        raise ValueError("schema_version must be 1")
    name = config.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")

    system = _mapping(config.get("system"), "system")
    _check_fields(system, {"frequency_mhz", "pdk", "flow"}, "system")
    frequency_mhz = _positive_number(
        system.get("frequency_mhz"), "system.frequency_mhz"
    )
    pdk = system.get("pdk")
    if not isinstance(pdk, str) or not pdk:
        raise ValueError("system.pdk must be a non-empty string")
    flow = system.get("flow", DEFAULT_FLOW)
    if not isinstance(flow, str) or not flow:
        raise ValueError("system.flow must be a non-empty string")

    compute = _mapping(config.get("compute"), "compute")
    _check_fields(compute, {"vector", "tensor"}, "compute")
    vector = _mapping(compute.get("vector"), "compute.vector")
    _check_fields(
        vector,
        {
            "fp32_elements_per_cycle",
            "utilization_pct",
            "implementation_variant",
        },
        "compute.vector",
    )
    tensor = _mapping(compute.get("tensor"), "compute.tensor")
    _check_fields(tensor, {"array_dim", "utilization_pct"}, "compute.tensor")
    memory = _mapping(config.get("memory"), "memory")
    _check_fields(memory, {"sram"}, "memory")
    sram = _mapping(memory.get("sram"), "memory.sram")
    _check_fields(
        sram,
        {
            "capacity_bytes",
            "bank_count",
            "bank_width_bytes",
            "layout_utilization_pct",
            "physical_macro_count",
            "port_groups",
            "simultaneous_read_write",
        },
        "memory.sram",
    )
    simultaneous_read_write = sram.get("simultaneous_read_write", False)
    if not isinstance(simultaneous_read_write, bool):
        raise ValueError(
            "memory.sram.simultaneous_read_write must be a boolean"
        )
    port_groups = sram.get("port_groups", [])
    if not isinstance(port_groups, list):
        raise ValueError("memory.sram.port_groups must be a list")
    normalized_port_groups = []
    for index, group_value in enumerate(port_groups):
        path = f"memory.sram.port_groups[{index}]"
        group = _mapping(group_value, path)
        _check_fields(
            group,
            {"name", "module_count", "read_widths_bytes", "write_widths_bytes"},
            path,
        )
        group_name = group.get("name")
        if not isinstance(group_name, str) or not group_name:
            raise ValueError(f"{path}.name must be a non-empty string")
        read_widths = group.get("read_widths_bytes")
        write_widths = group.get("write_widths_bytes")
        if not isinstance(read_widths, list) or not read_widths:
            raise ValueError(f"{path}.read_widths_bytes must be non-empty")
        if not isinstance(write_widths, list) or not write_widths:
            raise ValueError(f"{path}.write_widths_bytes must be non-empty")
        normalized_port_groups.append(
            {
                "name": group_name,
                "module_count": _positive_int(
                    group.get("module_count"), f"{path}.module_count"
                ),
                "read_widths_bytes": [
                    _positive_int(width, f"{path}.read_widths_bytes")
                    for width in read_widths
                ],
                "write_widths_bytes": [
                    _positive_int(width, f"{path}.write_widths_bytes")
                    for width in write_widths
                ],
            }
        )
    bank_count = sram.get("bank_count")
    bank_width_bytes = sram.get("bank_width_bytes")
    layout_utilization_pct = _positive_number(
        sram.get("layout_utilization_pct", 100.0),
        "memory.sram.layout_utilization_pct",
    )
    if layout_utilization_pct > 100.0:
        raise ValueError(
            "memory.sram.layout_utilization_pct must be at most 100"
        )
    if bank_count is not None or bank_width_bytes is not None:
        if port_groups:
            raise ValueError(
                "memory.sram fixed banks and port_groups are mutually exclusive"
            )
        bank_count = _positive_int(bank_count, "memory.sram.bank_count")
        bank_width_bytes = _positive_int(
            bank_width_bytes, "memory.sram.bank_width_bytes"
        )

    return {
        "schema_version": 1,
        "name": name,
        "system": {
            "frequency_mhz": frequency_mhz,
            "pdk": pdk,
            "flow": flow,
        },
        "compute": {
            "vector": {
                "fp32_elements_per_cycle": _positive_number(
                    vector.get("fp32_elements_per_cycle"),
                    "compute.vector.fp32_elements_per_cycle",
                ),
                "utilization_pct": _positive_number(
                    vector.get("utilization_pct", DEFAULT_UTILIZATION_PCT),
                    "compute.vector.utilization_pct",
                ),
                "implementation_variant": vector.get(
                    "implementation_variant", DEFAULT_ARA_VARIANT
                ),
            },
            "tensor": {
                "array_dim": _positive_number(
                    tensor.get("array_dim"), "compute.tensor.array_dim"
                ),
                "utilization_pct": _positive_number(
                    tensor.get("utilization_pct", DEFAULT_UTILIZATION_PCT),
                    "compute.tensor.utilization_pct",
                ),
            },
        },
        "memory": {
            "sram": {
                "capacity_bytes": _positive_int(
                    sram.get("capacity_bytes"),
                    "memory.sram.capacity_bytes",
                ),
                "port_groups": normalized_port_groups,
                "bank_count": bank_count,
                "bank_width_bytes": bank_width_bytes,
                "layout_utilization_pct": layout_utilization_pct,
                "physical_macro_count": (
                    _positive_int(
                        sram.get("physical_macro_count"),
                        "memory.sram.physical_macro_count",
                    )
                    if sram.get("physical_macro_count") is not None
                    else None
                ),
                "simultaneous_read_write": simultaneous_read_write,
            }
        },
    }


def _close(left, right):
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)


def select_record(
    records,
    *,
    pdk,
    flow,
    design,
    utilization_pct,
    system_period_ns,
    variant=None,
):
    matching = [
        record
        for record in records
        if record.pdk == pdk
        and record.flow == flow
        and record.design == design
        and _close(record.utilization_pct, utilization_pct)
        and (variant is None or record.variant == variant)
    ]
    if not matching:
        raise ValueError(
            "No datasheet records match "
            f"pdk={pdk}, flow={flow}, design={design}, "
            f"utilization_pct={utilization_pct}, variant={variant}"
        )
    frequency_capable = [
        record
        for record in matching
        if record.target_period_ns <= system_period_ns + 1e-9
    ]
    if not frequency_capable:
        periods = sorted({record.target_period_ns for record in matching})
        raise ValueError(
            f"No {design} record can target period <= {system_period_ns} ns; "
            f"available periods: {periods}"
        )

    timing_met = [record for record in frequency_capable if record.timing_met]
    timing_fallback = not timing_met
    pool = timing_met or frequency_capable
    closest_period = max(record.target_period_ns for record in pool)
    closest = [
        record
        for record in pool
        if _close(record.target_period_ns, closest_period)
    ]
    best_slack = max(record.timing_slack_ns for record in closest)
    best = [
        record
        for record in closest
        if _close(record.timing_slack_ns, best_slack)
    ]
    selected = sorted(best, key=lambda record: record.record_id)[0]
    return selected, {
        "timing_fallback": timing_fallback,
        "matching_record_count": len(matching),
        "frequency_capable_record_count": len(frequency_capable),
        "timing_met_record_count": len(timing_met),
        "best_record_tie_count": len(best),
        "warning": (
            "No timing-passing record matched; selected the closest target "
            "period with the greatest timing slack."
            if timing_fallback
            else ""
        ),
    }


def _module_result(
    kind,
    requested_ops_per_cycle,
    instance_ops_per_cycle,
    instance_count,
    record,
    selection,
    frequency_mhz,
):
    return {
        "kind": kind,
        "requested_ops_per_cycle": requested_ops_per_cycle,
        "instance_ops_per_cycle": instance_ops_per_cycle,
        "requested_tops": (
            requested_ops_per_cycle * frequency_mhz / 1_000_000.0
        ),
        "instance_tops": instance_ops_per_cycle * frequency_mhz / 1_000_000.0,
        "instance_count": instance_count,
        "provisioned_ops_per_cycle": (instance_count * instance_ops_per_cycle),
        "selected_record": asdict(record),
        "selection": selection,
        "power_w": instance_count * record.power_w,
        "area_um2": instance_count * record.area_um2,
        "area_mm2": instance_count * record.area_mm2,
    }


def _vector_result(
    requested_ops_per_cycle,
    instance_ops_per_cycle,
    instance_count,
    record,
    selection,
    frequency_mhz,
):
    shared_power_w = record.shared_power_w
    shared_area_um2 = record.shared_area_um2
    lane_group_power_w = record.power_w - shared_power_w
    lane_group_area_um2 = record.area_um2 - shared_area_um2
    if (
        shared_power_w <= 0
        or shared_area_um2 <= 0
        or lane_group_power_w <= 0
        or lane_group_area_um2 <= 0
    ):
        raise ValueError(
            f"ara_sys record {record.record_id} cannot be decomposed into "
            "positive shared and lane-group PPA"
        )
    power_w = shared_power_w + instance_count * lane_group_power_w
    area_um2 = shared_area_um2 + instance_count * lane_group_area_um2
    return {
        "kind": "vector",
        "requested_ops_per_cycle": requested_ops_per_cycle,
        "instance_ops_per_cycle": instance_ops_per_cycle,
        "requested_tops": (
            requested_ops_per_cycle * frequency_mhz / 1_000_000.0
        ),
        "instance_tops": instance_ops_per_cycle * frequency_mhz / 1_000_000.0,
        "instance_count": instance_count,
        "provisioned_ops_per_cycle": instance_count * instance_ops_per_cycle,
        "selected_record": asdict(record),
        "selection": selection,
        "scaling_model": "one_shared_cpu_and_other_plus_lane_groups",
        "shared_instance_count": 1,
        "shared_power_w": shared_power_w,
        "shared_area_um2": shared_area_um2,
        "shared_area_mm2": shared_area_um2 / 1_000_000.0,
        "lane_group_count": instance_count,
        "lane_group_lanes": ARA_LANES,
        "lane_group_power_w": lane_group_power_w,
        "lane_group_area_um2": lane_group_area_um2,
        "lane_group_area_mm2": lane_group_area_um2 / 1_000_000.0,
        "power_w": power_w,
        "area_um2": area_um2,
        "area_mm2": area_um2 / 1_000_000.0,
    }


def _sram_result(config, records, frequency_mhz):
    matching = [
        record
        for record in records
        if record.pdk == config["system"]["pdk"]
        and record.design == SRAM_DESIGN
    ]
    if len(matching) != 1:
        raise ValueError(
            "Expected exactly one SRAM record for "
            f"pdk={config['system']['pdk']}, design={SRAM_DESIGN}; "
            f"found {len(matching)}"
        )
    record = matching[0]
    if frequency_mhz > record.max_frequency_mhz + 1e-9:
        raise ValueError(
            f"SRAM {record.record_id} supports at most "
            f"{record.max_frequency_mhz:g} MHz, requested "
            f"{frequency_mhz:g} MHz"
        )
    requested = config["memory"]["sram"]["capacity_bytes"]
    capacity_instance_count = math.ceil(requested / record.capacity_bytes)
    word_bytes = record.word_bits // 8
    port_groups = []
    bandwidth_instance_count = 0
    required_bytes_per_cycle = 0
    required_read_bytes_per_cycle = 0
    required_write_bytes_per_cycle = 0
    simultaneous_read_write = config["memory"]["sram"].get(
        "simultaneous_read_write", False
    )
    bank_count = config["memory"]["sram"].get("bank_count")
    bank_width_bytes = config["memory"]["sram"].get("bank_width_bytes")
    if bank_count is not None:
        bandwidth_instance_count = bank_count
        required_read_bytes_per_cycle = bank_count * bank_width_bytes
        required_write_bytes_per_cycle = (
            required_read_bytes_per_cycle if simultaneous_read_write else 0
        )
        required_bytes_per_cycle = (
            required_read_bytes_per_cycle + required_write_bytes_per_cycle
        )
    for group in config["memory"]["sram"].get("port_groups", []):
        read_instances_per_module = sum(
            math.ceil(width / word_bytes)
            for width in group["read_widths_bytes"]
        )
        write_instances_per_module = sum(
            math.ceil(width / word_bytes)
            for width in group["write_widths_bytes"]
        )
        instances_per_module = (
            max(read_instances_per_module, write_instances_per_module)
            if simultaneous_read_write
            else read_instances_per_module + write_instances_per_module
        )
        instance_count = group["module_count"] * instances_per_module
        read_bytes_per_cycle = group["module_count"] * sum(
            group["read_widths_bytes"]
        )
        write_bytes_per_cycle = group["module_count"] * sum(
            group["write_widths_bytes"]
        )
        bytes_per_cycle = read_bytes_per_cycle + write_bytes_per_cycle
        bandwidth_instance_count += instance_count
        required_bytes_per_cycle += bytes_per_cycle
        required_read_bytes_per_cycle += read_bytes_per_cycle
        required_write_bytes_per_cycle += write_bytes_per_cycle
        port_groups.append(
            {
                **group,
                "read_instances_per_module": read_instances_per_module,
                "write_instances_per_module": write_instances_per_module,
                "instances_per_module": instances_per_module,
                "instance_count": instance_count,
                "required_bytes_per_cycle": bytes_per_cycle,
            }
        )
    physical_macro_count = config["memory"]["sram"].get(
        "physical_macro_count"
    )
    if (
        physical_macro_count is not None
        and physical_macro_count < capacity_instance_count
    ):
        raise ValueError(
            "memory.sram.physical_macro_count cannot provide the requested "
            "capacity"
        )
    instance_count = (
        physical_macro_count
        if physical_macro_count is not None
        else max(capacity_instance_count, bandwidth_instance_count)
    )
    macro_area_um2 = instance_count * record.area_um2
    layout_utilization_pct = config["memory"]["sram"].get(
        "layout_utilization_pct", 100.0
    )
    floorplan_area_um2 = macro_area_um2 / (layout_utilization_pct / 100.0)
    return {
        "kind": "sram",
        "power_kind": "reported",
        "requested_capacity_bytes": requested,
        "instance_capacity_bytes": record.capacity_bytes,
        "instance_word_bytes": word_bytes,
        "capacity_instance_count": capacity_instance_count,
        "bandwidth_instance_count": bandwidth_instance_count,
        "physical_macro_count_override": physical_macro_count,
        "instance_count": instance_count,
        "provisioned_capacity_bytes": (instance_count * record.capacity_bytes),
        "required_bytes_per_cycle": required_bytes_per_cycle,
        "required_read_bytes_per_cycle": required_read_bytes_per_cycle,
        "required_write_bytes_per_cycle": required_write_bytes_per_cycle,
        "simultaneous_read_write": simultaneous_read_write,
        "bank_count": bank_count,
        "bank_width_bytes": bank_width_bytes,
        "port_groups": port_groups,
        "selected_record": asdict(record),
        "power_w": instance_count * record.power_w,
        "layout_utilization_pct": layout_utilization_pct,
        "macro_area_um2": macro_area_um2,
        "macro_area_mm2": macro_area_um2 / 1_000_000.0,
        "floorplan_area_um2": floorplan_area_um2,
        "floorplan_area_mm2": floorplan_area_um2 / 1_000_000.0,
        "area_um2": floorplan_area_um2,
        "area_mm2": floorplan_area_um2 / 1_000_000.0,
    }


def evaluate_config(
    config,
    records,
    datasheet_path,
    sram_records,
    sram_datasheet_path,
):
    frequency_mhz = config["system"]["frequency_mhz"]
    system_period_ns = 1000.0 / frequency_mhz
    pdk = config["system"]["pdk"]
    flow = config["system"]["flow"]
    sram = _sram_result(config, sram_records, frequency_mhz)

    vector_config = config["compute"]["vector"]
    vector_elements = vector_config["fp32_elements_per_cycle"]
    ara_elements = ARA_LANES * ARA_FP32_ELEMENTS_PER_LANE_PER_CYCLE
    vector_ops = vector_elements * OPS_PER_FMA
    ara_ops = ara_elements * OPS_PER_FMA
    vector_count = math.ceil(vector_elements / ara_elements)
    vector_record, vector_selection = select_record(
        records,
        pdk=pdk,
        flow=flow,
        design="ara_sys",
        utilization_pct=vector_config["utilization_pct"],
        system_period_ns=system_period_ns,
        variant=vector_config["implementation_variant"],
    )
    vector = _vector_result(
        vector_ops,
        ara_ops,
        vector_count,
        vector_record,
        vector_selection,
        frequency_mhz,
    )
    vector.update(
        {
            "requested_fp32_elements_per_cycle": vector_elements,
            "instance_fp32_elements_per_cycle": ara_elements,
            "op_counting": "fp32_fma_as_two_ops",
        }
    )

    tensor_config = config["compute"]["tensor"]
    tensor_macs = tensor_config["array_dim"] ** 2
    mesh_macs = MESH_DIM**2
    tensor_ops = tensor_macs * OPS_PER_FMA
    mesh_ops = mesh_macs * OPS_PER_FMA
    tensor_count = math.ceil(tensor_ops / mesh_ops)
    tensor_record, tensor_selection = select_record(
        records,
        pdk=pdk,
        flow=flow,
        design="Mesh_BOTH_32x32",
        utilization_pct=tensor_config["utilization_pct"],
        system_period_ns=system_period_ns,
    )
    tensor = _module_result(
        "tensor",
        tensor_ops,
        mesh_ops,
        tensor_count,
        tensor_record,
        tensor_selection,
        frequency_mhz,
    )
    tensor.update(
        {
            "requested_macs_per_cycle": tensor_macs,
            "instance_macs_per_cycle": mesh_macs,
            "op_counting": "multiply_add_as_two_ops",
        }
    )
    compute_power_w = vector["power_w"] + tensor["power_w"]
    compute_area_um2 = vector["area_um2"] + tensor["area_um2"]
    compute_area_mm2 = vector["area_mm2"] + tensor["area_mm2"]

    warnings = [
        module["selection"]["warning"]
        for module in (vector, tensor)
        if module["selection"]["warning"]
    ]
    return {
        "schema_version": 1,
        "name": config["name"],
        "system": {
            **config["system"],
            "period_ns": system_period_ns,
        },
        "datasheet": {
            "compute": {
                "path": str(datasheet_path),
                "sha256": _sha256(datasheet_path),
            },
            "sram": {
                "path": str(sram_datasheet_path),
                "sha256": _sha256(sram_datasheet_path),
            },
        },
        "modules": {"vector": vector, "tensor": tensor, "sram": sram},
        "totals": {
            "compute_power_w": compute_power_w,
            "sram_power_w": sram["power_w"],
            "power_w": compute_power_w + sram["power_w"],
            "compute_area_um2": compute_area_um2,
            "compute_area_mm2": compute_area_mm2,
            "sram_area_um2": sram["area_um2"],
            "sram_area_mm2": sram["area_mm2"],
            "area_um2": compute_area_um2 + sram["area_um2"],
            "area_mm2": compute_area_mm2 + sram["area_mm2"],
        },
        "warnings": warnings,
    }


def compare_results(baseline, candidate):
    baseline_power = baseline["totals"]["power_w"]
    baseline_area = baseline["totals"]["area_mm2"]
    if baseline_power == 0 or baseline_area == 0:
        raise ValueError("Baseline Power and Area must be non-zero")
    candidate_power = candidate["totals"]["power_w"]
    candidate_area = candidate["totals"]["area_mm2"]
    return {
        "baseline": baseline,
        "candidate": candidate,
        "comparison": {
            "power_ratio": candidate_power / baseline_power,
            "area_ratio": candidate_area / baseline_area,
            "power_improvement_pct": (
                (baseline_power - candidate_power) / baseline_power * 100.0
            ),
            "area_improvement_pct": (
                (baseline_area - candidate_area) / baseline_area * 100.0
            ),
        },
    }
