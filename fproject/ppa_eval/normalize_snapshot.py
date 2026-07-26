#!/usr/bin/env python3

# Copyright (c) 2026
# All rights reserved.

import argparse
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree


NAMESPACE = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
}
FIELDNAMES = (
    "record_id",
    "source_kind",
    "source_file_sha256",
    "source_sheet",
    "source_row",
    "pdk",
    "raw_pdk",
    "flow",
    "raw_flow",
    "design",
    "variant",
    "utilization_pct",
    "target_period_ns",
    "timing_period_ns",
    "timing_slack_ns",
    "timing_met",
    "raw_timing",
    "area_um2",
    "area_mm2",
    "power_w",
    "shared_power_w",
    "shared_area_um2",
    "shared_area_mm2",
    "shared_derived_from",
    "notes",
    "derived_from",
    "power_scale",
    "area_scale",
)
VALID_DESIGNS = {"ara_sys", "Mesh_BOTH_32x32"}
ARA_SHARED_DESIGNS = {"cpu", "other"}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _column_number(cell_reference):
    letters = re.match(r"[A-Z]+", cell_reference).group(0)
    number = 0
    for letter in letters:
        number = number * 26 + ord(letter) - ord("A") + 1
    return number


def _shared_strings(archive):
    try:
        payload = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ElementTree.fromstring(payload)
    strings = []
    for item in root.findall("main:si", NAMESPACE):
        strings.append(
            "".join(
                node.text or ""
                for node in item.iterfind(".//main:t", NAMESPACE)
            )
        )
    return strings


def _cell_value(cell, shared_strings):
    value = cell.find("main:v", NAMESPACE)
    if value is None:
        inline = cell.find("main:is/main:t", NAMESPACE)
        return inline.text if inline is not None else ""
    if cell.get("t") == "s":
        return shared_strings[int(value.text)]
    return value.text or ""


def read_first_sheet_rows(path):
    with zipfile.ZipFile(path) as archive:
        shared_strings = _shared_strings(archive)
        root = ElementTree.fromstring(
            archive.read("xl/worksheets/sheet1.xml")
        )
    rows = []
    for row in root.findall(".//main:sheetData/main:row", NAMESPACE):
        values = {}
        for cell in row.findall("main:c", NAMESPACE):
            values[_column_number(cell.get("r"))] = _cell_value(
                cell, shared_strings
            ).strip()
        rows.append((int(row.get("r")), values))
    return rows


def normalize_flow(raw_flow):
    if "DC-Innovus" in raw_flow:
        return "DC-Innovus"
    return raw_flow


def parse_timing(raw_timing, target_period_ns):
    if "/" in raw_timing:
        period_text, slack_text = raw_timing.split("/", 1)
        period = float(period_text)
        slack = float(slack_text)
    else:
        period = target_period_ns
        slack = float(raw_timing)
    return period, slack


def infer_variant(design, notes):
    if design != "ara_sys":
        return "default"
    lowered = notes.lower()
    if "no macro" in lowered:
        return "no_macro"
    if "with macro" in lowered:
        return "with_macro"
    return "unspecified"


def normalize_snapshot(path):
    source_sha256 = _sha256(path)
    source_rows = read_first_sheet_rows(path)
    shared_rows = [
        (row_number, cells)
        for row_number, cells in source_rows
        if cells.get(3, "").strip().lower() in ARA_SHARED_DESIGNS
    ]
    shared_power_w = sum(float(cells.get(11, 0)) for _, cells in shared_rows)
    shared_area_um2 = sum(float(cells.get(8, 0)) for _, cells in shared_rows)
    shared_derived_from = ",".join(
        f"source-sheet1-r{row_number}" for row_number, _ in shared_rows
    )
    records = []
    ignored = []
    for row_number, cells in source_rows:
        raw_pdk = cells.get(1, "")
        raw_flow = cells.get(2, "")
        design = cells.get(3, "")
        utilization = cells.get(4, "")
        period = cells.get(5, "")
        area = cells.get(8, "")
        raw_timing = cells.get(10, "")
        power = cells.get(11, "")
        notes = cells.get(13, "")

        if design not in VALID_DESIGNS:
            ignored.append(
                {"row": row_number, "reason": "not a PPA design row"}
            )
            continue
        required = {
            "pdk": raw_pdk,
            "flow": raw_flow,
            "utilization": utilization,
            "period": period,
            "area": area,
            "timing": raw_timing,
            "power": power,
        }
        missing = [key for key, value in required.items() if value == ""]
        if missing:
            ignored.append(
                {
                    "row": row_number,
                    "reason": f"missing fields: {', '.join(missing)}",
                }
            )
            continue

        target_period_ns = float(period)
        timing_period_ns, timing_slack_ns = parse_timing(
            raw_timing, target_period_ns
        )
        area_um2 = float(area)
        record = {
                "record_id": f"source-sheet1-r{row_number}",
                "source_kind": "source",
                "source_file_sha256": source_sha256,
                "source_sheet": "Sheet1",
                "source_row": str(row_number),
                "pdk": raw_pdk,
                "raw_pdk": raw_pdk,
                "flow": normalize_flow(raw_flow),
                "raw_flow": raw_flow,
                "design": design,
                "variant": infer_variant(design, notes),
                "utilization_pct": float(utilization) * 100.0,
                "target_period_ns": target_period_ns,
                "timing_period_ns": timing_period_ns,
                "timing_slack_ns": timing_slack_ns,
                "timing_met": timing_slack_ns >= 0.0,
                "raw_timing": raw_timing,
                "area_um2": area_um2,
                "area_mm2": area_um2 / 1_000_000.0,
                "power_w": float(power),
                "shared_power_w": "",
                "shared_area_um2": "",
                "shared_area_mm2": "",
                "shared_derived_from": "",
                "notes": notes,
                "derived_from": "",
                "power_scale": 1.0,
                "area_scale": 1.0,
            }
        if design == "ara_sys" and raw_pdk == "T7":
            record.update(
                {
                    "shared_power_w": shared_power_w,
                    "shared_area_um2": shared_area_um2,
                    "shared_area_mm2": shared_area_um2 / 1_000_000.0,
                    "shared_derived_from": shared_derived_from,
                }
            )
        records.append(record)
    return records, ignored


def write_records(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output, fieldnames=FIELDNAMES, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    records, ignored = normalize_snapshot(args.input)
    write_records(records, args.output)
    audit = {
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "record_count": len(records),
        "ignored": ignored,
    }
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
