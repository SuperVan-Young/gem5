# Copyright (c) 2026
# All rights reserved.

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import m5
from m5.objects import AddrRange

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "configs"))

from npu_test_system import (  # noqa: E402
    DEFAULT_NPU_MMIO_BUS_KWARGS,
    CacheHierarchyConfig,
    NPUTestSystemBuilder,
)

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
parser.add_argument("--membus-width", type=int)
parser.add_argument("--membus-frontend-latency", type=int)
parser.add_argument("--membus-forward-latency", type=int)
parser.add_argument("--membus-response-latency", type=int)
parser.add_argument("--membus-header-latency", type=int)
parser.add_argument("--membus-snoop-response-latency", type=int)
parser.add_argument("--membus-snoop-filter-lookup-latency", type=int)
parser.add_argument("--enable-l1-caches", action="store_true")
parser.add_argument("--disable-l1-caches", action="store_true")
parser.add_argument("--enable-l2-cache", action="store_true")
parser.add_argument("--l1i-size")
parser.add_argument("--l1d-size")
parser.add_argument("--l2-size")
args = parser.parse_args()

EXPECTED_EXIT_CAUSE = "exiting with last active thread context"
EXPECTED_EXIT_CODE = 0
DEFAULT_MEMBUS_CONFIG = {
    "width": 64,
    "frontend_latency": 0,
    "forward_latency": 0,
    "response_latency": 1,
    "header_latency": 1,
    "snoop_response_latency": 0,
    "snoop_filter_lookup_latency": 0,
}
DEFAULT_CACHE_CONFIG = CacheHierarchyConfig()

binary = os.path.abspath(args.binary)
enable_l1_caches = True
if args.disable_l1_caches:
    enable_l1_caches = False
elif args.enable_l1_caches:
    enable_l1_caches = True
membus_kwargs = {
    "width": (
        DEFAULT_MEMBUS_CONFIG["width"]
        if args.membus_width is None
        else args.membus_width
    ),
    "frontend_latency": (
        DEFAULT_MEMBUS_CONFIG["frontend_latency"]
        if args.membus_frontend_latency is None
        else args.membus_frontend_latency
    ),
    "forward_latency": (
        DEFAULT_MEMBUS_CONFIG["forward_latency"]
        if args.membus_forward_latency is None
        else args.membus_forward_latency
    ),
    "response_latency": (
        DEFAULT_MEMBUS_CONFIG["response_latency"]
        if args.membus_response_latency is None
        else args.membus_response_latency
    ),
    "header_latency": (
        DEFAULT_MEMBUS_CONFIG["header_latency"]
        if args.membus_header_latency is None
        else args.membus_header_latency
    ),
    "snoop_response_latency": (
        DEFAULT_MEMBUS_CONFIG["snoop_response_latency"]
        if args.membus_snoop_response_latency is None
        else args.membus_snoop_response_latency
    ),
}
cache_config = CacheHierarchyConfig(
    enable_l1=enable_l1_caches,
    enable_l2=args.enable_l2_cache,
    l1i=replace(
        DEFAULT_CACHE_CONFIG.l1i,
        size=(
            DEFAULT_CACHE_CONFIG.l1i.size
            if args.l1i_size is None
            else args.l1i_size
        ),
    ),
    l1d=replace(
        DEFAULT_CACHE_CONFIG.l1d,
        size=(
            DEFAULT_CACHE_CONFIG.l1d.size
            if args.l1d_size is None
            else args.l1d_size
        ),
    ),
    l2=replace(
        DEFAULT_CACHE_CONFIG.l2,
        size=(
            DEFAULT_CACHE_CONFIG.l2.size
            if args.l2_size is None
            else args.l2_size
        ),
    ),
)
builder = NPUTestSystemBuilder(
    clock="4GHz",
    mem_ranges=[
        AddrRange(0, size=0x60000000),
        AddrRange(0x60000000, size=64 * 1024),
    ],
    cache_config=cache_config,
)
system = builder.build_base_system(
    membus_kwargs=membus_kwargs,
    npu_mmio_bus_kwargs=DEFAULT_NPU_MMIO_BUS_KWARGS,
    cpu_npu_mmio_bus_kwargs=DEFAULT_NPU_MMIO_BUS_KWARGS,
)
system.membus.snoop_filter.lookup_latency = (
    DEFAULT_MEMBUS_CONFIG["snoop_filter_lookup_latency"]
    if args.membus_snoop_filter_lookup_latency is None
    else args.membus_snoop_filter_lookup_latency
)
builder.add_default_physmem()
builder.add_spm()
cpu = builder.add_cpu(cpu_id=0)
builder.set_workload(binary, cpu_id=0, pid=400)
builder.add_megacmdqueue()
lut = builder.add_lut(
    range_reduction_latency="20ns",
    lookup_latency="30ns",
    interpolation_latency="20ns",
    normalize_latency="20ns",
)
dma = builder.add_dma()
vpu0 = builder.add_vpu(
    vpu_id=0,
    num_mem_side_ports=4,
    debug_process_latency="200ns",
    lut=lut,
)
vpu1 = builder.add_vpu(
    vpu_id=1,
    num_mem_side_ports=4,
    debug_process_latency="200ns",
    lut=lut,
)
builder.instantiate_root()
m5.instantiate()

process = builder.get_process(0)
builder.map_cmdq(process=process)
builder.map_dram(process=process)
builder.map_spm(process=process)

exit_event = m5.simulate()
exit_cause = exit_event.getCause()
exit_code = exit_event.getCode()

cmdq_occupancy = system.cmdq.queueOccupancy()
dma_completed = dma.completedCmdCount()
vpu0_completed = vpu0.completedCmdCount()
vpu1_completed = vpu1.completedCmdCount()

print(f"PROFILE_TILE_EXIT_CAUSE={exit_cause}")
print(f"PROFILE_TILE_EXIT_CODE={exit_code}")
print(f"PROFILE_TILE_CMDQ_OCCUPANCY={cmdq_occupancy}")
print(f"PROFILE_TILE_DMA_COMPLETED={dma_completed}")
print(f"PROFILE_TILE_VPU0_COMPLETED={vpu0_completed}")
print(f"PROFILE_TILE_VPU1_COMPLETED={vpu1_completed}")
print(f"PROFILE_TILE_ENABLE_L1_CACHES={int(enable_l1_caches)}")
print(f"PROFILE_TILE_ENABLE_L2_CACHE={int(args.enable_l2_cache)}")
print(f"PROFILE_TILE_MEMBUS_WIDTH={system.membus.width}")
print(
    "PROFILE_TILE_MEMBUS_FRONTEND_LATENCY="
    f"{system.membus.frontend_latency}"
)
print(
    "PROFILE_TILE_MEMBUS_FORWARD_LATENCY="
    f"{system.membus.forward_latency}"
)
print(
    "PROFILE_TILE_MEMBUS_RESPONSE_LATENCY="
    f"{system.membus.response_latency}"
)
print(
    "PROFILE_TILE_MEMBUS_HEADER_LATENCY="
    f"{system.membus.header_latency}"
)
print(
    "PROFILE_TILE_MEMBUS_SNOOP_RESPONSE_LATENCY="
    f"{system.membus.snoop_response_latency}"
)
print(
    "PROFILE_TILE_MEMBUS_SNOOP_FILTER_LOOKUP_LATENCY="
    f"{system.membus.snoop_filter.lookup_latency}"
)
print(f"PROFILE_TILE_NPU_MMIO_BUS_WIDTH={system.npu_mmio_bus.width}")
print(
    "PROFILE_TILE_NPU_MMIO_BUS_FRONTEND_LATENCY="
    f"{system.npu_mmio_bus.frontend_latency}"
)
print(
    "PROFILE_TILE_NPU_MMIO_BUS_FORWARD_LATENCY="
    f"{system.npu_mmio_bus.forward_latency}"
)
print(
    "PROFILE_TILE_NPU_MMIO_BUS_RESPONSE_LATENCY="
    f"{system.npu_mmio_bus.response_latency}"
)
print(
    "PROFILE_TILE_NPU_MMIO_BUS_HEADER_LATENCY="
    f"{system.npu_mmio_bus.header_latency}"
)
if enable_l1_caches:
    print(f"PROFILE_TILE_L1I_SIZE={cpu.icache.size}")
    print(f"PROFILE_TILE_L1D_SIZE={cpu.dcache.size}")
if args.enable_l2_cache:
    print(f"PROFILE_TILE_L2_SIZE={system.l2cache.size}")

if (
    exit_cause == EXPECTED_EXIT_CAUSE
    and exit_code == EXPECTED_EXIT_CODE
    and cmdq_occupancy == 0
    and dma_completed >= 2
    and vpu0_completed >= 1
    and vpu1_completed >= 1
):
    print("PROFILE_TILE_CONFIG_PASS")
