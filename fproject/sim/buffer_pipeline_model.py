# Copyright (c) 2026
# All rights reserved.

def evaluate_buffer_pipeline(
    *,
    qk_busy_cycles,
    softmax_busy_cycles,
    pv_busy_cycles,
    bank_count,
    bank_width_bytes,
    banks_per_engine,
    buffer_slots,
):
    values = {
        "qk_busy_cycles": qk_busy_cycles,
        "softmax_busy_cycles": softmax_busy_cycles,
        "pv_busy_cycles": pv_busy_cycles,
        "bank_count": bank_count,
        "bank_width_bytes": bank_width_bytes,
        "banks_per_engine": banks_per_engine,
        "buffer_slots": buffer_slots,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    engine_bank_groups = bank_count // banks_per_engine
    if engine_bank_groups < 1:
        raise ValueError(
            "bank_count must provide at least one complete engine bank group"
        )

    non_overlap_cycles = (
        qk_busy_cycles + softmax_busy_cycles + pv_busy_cycles
    )
    overlap_enabled = engine_bank_groups >= 2 and buffer_slots >= 2
    stages = [
        {
            "name": "serial",
            "qk_cycles": qk_busy_cycles,
            "softmax_cycles": softmax_busy_cycles,
            "pv_cycles": pv_busy_cycles,
            "cycles": non_overlap_cycles,
        }
    ]
    if overlap_enabled:
        if buffer_slots != 2:
            raise ValueError("overlap model currently requires two buffer slots")
        qk_slots = _split_cycles(qk_busy_cycles)
        softmax_slots = _split_cycles(softmax_busy_cycles)
        pv_slots = _split_cycles(pv_busy_cycles)
        stages = [
            {
                "name": "qk0",
                "qk_cycles": qk_slots[0],
                "softmax_cycles": 0,
                "pv_cycles": 0,
                "cycles": qk_slots[0],
            },
            {
                "name": "qk1_softmax0",
                "qk_cycles": qk_slots[1],
                "softmax_cycles": softmax_slots[0],
                "pv_cycles": 0,
                "cycles": max(qk_slots[1], softmax_slots[0]),
            },
            {
                "name": "softmax1_pv0",
                "qk_cycles": 0,
                "softmax_cycles": softmax_slots[1],
                "pv_cycles": pv_slots[0],
                "cycles": max(softmax_slots[1], pv_slots[0]),
            },
            {
                "name": "pv1",
                "qk_cycles": 0,
                "softmax_cycles": 0,
                "pv_cycles": pv_slots[1],
                "cycles": pv_slots[1],
            },
        ]
    overlapped_cycles = sum(stage["cycles"] for stage in stages)
    saved_cycles = non_overlap_cycles - overlapped_cycles
    return {
        "bank_count": bank_count,
        "bank_width_bytes": bank_width_bytes,
        "bandwidth_bytes_per_cycle": bank_count * bank_width_bytes,
        "banks_per_engine": banks_per_engine,
        "engine_bandwidth_bytes_per_cycle": (
            banks_per_engine * bank_width_bytes
        ),
        "engine_bank_groups": engine_bank_groups,
        "buffer_slots": buffer_slots,
        "overlap_enabled": overlap_enabled,
        "qk_busy_cycles": qk_busy_cycles,
        "softmax_busy_cycles": softmax_busy_cycles,
        "pv_busy_cycles": pv_busy_cycles,
        "non_overlap_cycles": non_overlap_cycles,
        "overlapped_cycles": overlapped_cycles,
        "saved_cycles": saved_cycles,
        "stages": stages,
        "time_reduction_percent": (
            saved_cycles / non_overlap_cycles * 100.0
        ),
        "speedup": non_overlap_cycles / overlapped_cycles,
        "model": (
            "two_slot_br_pipeline"
            if overlap_enabled
            else "shared_bank_serial_execution"
        ),
    }


def _split_cycles(cycles):
    first = (cycles + 1) // 2
    return (first, cycles - first)
