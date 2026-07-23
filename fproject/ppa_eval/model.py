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
MESH_DIM = 32
DEFAULT_FLOW = "DC-Innovus"
DEFAULT_UTILIZATION_PCT = 50.0
DEFAULT_ARA_VARIANT = "no_macro"


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
        )


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


def load_config(path):
    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "root")
    _check_fields(
        config,
        {"schema_version", "name", "system", "compute", "simulation"},
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
            "fp32_ops_per_cycle",
            "utilization_pct",
            "implementation_variant",
        },
        "compute.vector",
    )
    tensor = _mapping(compute.get("tensor"), "compute.tensor")
    _check_fields(tensor, {"array_dim", "utilization_pct"}, "compute.tensor")

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
                "fp32_ops_per_cycle": _positive_number(
                    vector.get("fp32_ops_per_cycle"),
                    "compute.vector.fp32_ops_per_cycle",
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


def evaluate_config(config, records, datasheet_path):
    frequency_mhz = config["system"]["frequency_mhz"]
    system_period_ns = 1000.0 / frequency_mhz
    pdk = config["system"]["pdk"]
    flow = config["system"]["flow"]

    vector_config = config["compute"]["vector"]
    vector_ops = vector_config["fp32_ops_per_cycle"]
    vector_count = math.ceil(vector_ops / ARA_LANES)
    vector_record, vector_selection = select_record(
        records,
        pdk=pdk,
        flow=flow,
        design="ara_sys",
        utilization_pct=vector_config["utilization_pct"],
        system_period_ns=system_period_ns,
        variant=vector_config["implementation_variant"],
    )
    vector = _module_result(
        "vector",
        vector_ops,
        ARA_LANES,
        vector_count,
        vector_record,
        vector_selection,
        frequency_mhz,
    )

    tensor_config = config["compute"]["tensor"]
    tensor_ops = tensor_config["array_dim"] ** 2
    mesh_ops = MESH_DIM**2
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
            "path": str(datasheet_path),
            "sha256": _sha256(datasheet_path),
        },
        "modules": {"vector": vector, "tensor": tensor},
        "totals": {
            "power_w": vector["power_w"] + tensor["power_w"],
            "area_um2": vector["area_um2"] + tensor["area_um2"],
            "area_mm2": vector["area_mm2"] + tensor["area_mm2"],
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
