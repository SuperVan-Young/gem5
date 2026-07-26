#!/usr/bin/env python3

# Copyright (c) 2026
# All rights reserved.

import argparse
import csv
from pathlib import Path

try:
    from .normalize_snapshot import FIELDNAMES, write_records
except ImportError:
    from normalize_snapshot import FIELDNAMES, write_records


def read_records(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _scaled(value, scale):
    return float(value) * scale


def augment_f7_ara(
    records,
    source_pdk="T7",
    target_pdk="F7-PKU",
    power_scale=0.50,
    area_scale=0.75,
):
    if any(
        record["pdk"] == target_pdk and record["design"] == "ara_sys"
        for record in records
    ):
        raise ValueError(
            f"{target_pdk} already contains ara_sys records; refusing to "
            "overwrite real data"
        )

    augmented = []
    for record in records:
        if record["pdk"] != source_pdk or record["design"] != "ara_sys":
            continue
        clone = dict(record)
        clone["record_id"] = f"aug-{target_pdk.lower()}-{record['record_id']}"
        clone["source_kind"] = "augmented"
        clone["pdk"] = target_pdk
        clone["raw_pdk"] = target_pdk
        clone["raw_flow"] = f"{target_pdk}-DC-Innovus"
        clone["derived_from"] = record["record_id"]
        clone["power_scale"] = power_scale
        clone["area_scale"] = area_scale
        clone["power_w"] = _scaled(record["power_w"], power_scale)
        clone["area_um2"] = _scaled(record["area_um2"], area_scale)
        clone["area_mm2"] = float(clone["area_um2"]) / 1_000_000.0
        clone["shared_power_w"] = _scaled(
            record["shared_power_w"], power_scale
        )
        clone["shared_area_um2"] = _scaled(
            record["shared_area_um2"], area_scale
        )
        clone["shared_area_mm2"] = (
            float(clone["shared_area_um2"]) / 1_000_000.0
        )
        clone["notes"] = (
            f"augmented from {record['record_id']}: "
            f"power_scale={power_scale}, area_scale={area_scale}; "
            f"{record['notes']}"
        )
        augmented.append(clone)
    if not augmented:
        raise ValueError(f"No {source_pdk} ara_sys source records found")
    return [*records, *augmented]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-pdk", default="T7")
    parser.add_argument("--target-pdk", default="F7-PKU")
    parser.add_argument("--power-scale", type=float, default=0.50)
    parser.add_argument("--area-scale", type=float, default=0.75)
    return parser.parse_args()


def main():
    args = parse_args()
    records = read_records(args.input)
    unexpected = set(records[0]) - set(FIELDNAMES) if records else set()
    if unexpected:
        raise ValueError(f"Unexpected input fields: {sorted(unexpected)}")
    augmented = augment_f7_ara(
        records,
        source_pdk=args.source_pdk,
        target_pdk=args.target_pdk,
        power_scale=args.power_scale,
        area_scale=args.area_scale,
    )
    write_records(augmented, args.output)
    print(
        f"Wrote {len(augmented)} records "
        f"({len(augmented) - len(records)} augmented) to {args.output}"
    )


if __name__ == "__main__":
    main()
