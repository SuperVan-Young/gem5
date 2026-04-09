# Copyright (c) 2026
# All rights reserved.

import argparse
import json
from pathlib import Path


MARKER = "NPU_PROFILE "
SCHEMA = "npu-profile-v1"


def _load_profile_records(log_path: Path):
    records = []
    for line_no, line in enumerate(log_path.read_text().splitlines(), start=1):
        marker_index = line.find(MARKER)
        if marker_index < 0:
            continue
        payload = line[marker_index + len(MARKER) :].strip()
        json_start = payload.find("{")
        if json_start < 0:
            raise ValueError(
                f"{log_path}:{line_no}: missing JSON payload after {MARKER!r}"
            )
        try:
            record = json.loads(payload[json_start:])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{log_path}:{line_no}: invalid profiling JSON: {exc}"
            ) from exc
        record["_line"] = line_no
        records.append(record)
    if not records:
        raise ValueError(f"{log_path}: no profiling records found")
    return records


def _axis_display(event):
    return f"{event['seu_type']}[{event['device_id']}]"


def _event_title(begin):
    details = begin.get("details", {})
    operation = (
        details.get("opcode")
        or details.get("mode")
        or begin["macro_kind"]
    )
    return (
        f"{_axis_display(begin)} macro {begin['macro_id']} "
        f"{operation}"
    )


def _pair_records(records):
    active = {}
    paired = []

    for record in records:
        key = (record["seu_name"], int(record["macro_id"]))
        phase = record["event"]

        if phase == "begin":
            if key in active:
                prev = active[key]
                raise ValueError(
                    "duplicate begin event for "
                    f"{key!r} at lines {prev['_line']} and {record['_line']}"
                )
            active[key] = record
            continue

        if phase != "end":
            raise ValueError(
                "unsupported profiling event "
                f"{phase!r} at line {record['_line']}"
            )

        if key not in active:
            raise ValueError(
                f"end event without begin for {key!r} "
                f"at line {record['_line']}"
            )

        begin = active.pop(key)
        if int(record["tick"]) < int(begin["tick"]):
            raise ValueError(
                "end tick precedes begin tick for "
                f"{key!r}: {begin['tick']} -> {record['tick']}"
            )

        paired.append(
            {
                "axis": begin["seu_name"],
                "axis_display": _axis_display(begin),
                "title": _event_title(begin),
                "seu_name": begin["seu_name"],
                "seu_type": begin["seu_type"],
                "device_id": begin["device_id"],
                "macro_id": begin["macro_id"],
                "issue_queue": begin["issue_queue"],
                "macro_kind": begin["macro_kind"],
                "start_tick": begin["tick"],
                "end_tick": record["tick"],
                "duration": int(record["tick"]) - int(begin["tick"]),
                "raw_words": begin.get("raw_words", []),
                "details": begin.get("details", {}),
                "begin": begin,
                "end": record,
            }
        )

    if active:
        first_key = next(iter(active))
        dangling = active[first_key]
        raise ValueError(
            f"missing end event for {first_key!r} "
            f"begun at line {dangling['_line']}"
        )

    paired.sort(
        key=lambda event: (
            event["start_tick"],
            event["axis"],
            event["macro_id"],
        )
    )

    axis_meta = []
    axis_index = {}
    for event in paired:
        axis = event["axis"]
        if axis in axis_index:
            continue
        axis_index[axis] = len(axis_meta)
        axis_meta.append(
            {
                "axis": axis,
                "axis_display": event["axis_display"],
                "seu_type": event["seu_type"],
                "device_id": event["device_id"],
            }
        )

    per_axis_order = {}
    for event in paired:
        axis = event["axis"]
        order = per_axis_order.get(axis, 0)
        per_axis_order[axis] = order + 1
        event["axis_order"] = order

    return axis_meta, paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="gem5 debug log path")
    parser.add_argument(
        "--output",
        required=True,
        help="parsed JSON output path",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    records = _load_profile_records(input_path)
    axes, events = _pair_records(records)

    payload = {
        "schema": SCHEMA,
        "source": str(input_path),
        "axis_count": len(axes),
        "event_count": len(events),
        "axes": axes,
        "events": events,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
