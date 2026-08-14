# Copyright (c) 2026
# All rights reserved.

import os
import sys
from pathlib import Path

from m5.objects import AddrRange

REPO_ROOT = Path(__file__).resolve().parents[2]
NPU_CONFIGS = REPO_ROOT / "tests/gem5/npu/configs"
sys.path.insert(0, str(NPU_CONFIGS))

from npu_test_system import (  # noqa: E402
    NPUAddressMap,
    NPUTestSystemBuilder,
)


def build_top(config, binary, workload_args):
    interconnect = config["interconnect"]
    dram = config["memory"]["dram"]
    spm = config["memory"]["spm"]
    mpu = config["mpu"]
    vector = config["vector"]
    lut_config = config["lut"]

    # Base system and the buses shared by the CPU, memory, and NPU.
    bus_latency = interconnect["latency_cycles"]
    bus_options = {
        "width": interconnect["width_bytes"],
        "frontend_latency": bus_latency,
        "forward_latency": bus_latency,
        "response_latency": bus_latency,
        "header_latency": bus_latency,
    }
    builder = NPUTestSystemBuilder(
        clock=f"{config['clock_mhz']}MHz",
        mem_ranges=[
            AddrRange(0, size=spm["base_address"]),
            AddrRange(spm["base_address"], size=spm["size_bytes"]),
        ],
        addr_map=NPUAddressMap(spm_base=spm["base_address"]),
    )
    builder.build_base_system(
        membus_kwargs=bus_options,
        npu_mmio_bus_kwargs=bus_options,
        cpu_npu_mmio_bus_kwargs=bus_options,
    )

    # Main memory and the NPU scratchpad.
    builder.add_lowmem(
        AddrRange(0, size=spm["base_address"]),
        latency=dram["latency"],
        bandwidth=dram["bandwidth"],
    )
    builder.add_spm(
        base_addr=spm["base_address"],
        size=spm["size_bytes"],
        latency=spm["latency"],
        bandwidth=spm["bandwidth"],
        pipeline_depth=spm["pipeline_depth"],
        pipeline_ports=spm["pipeline_ports"],
        pipeline_port_stride=spm["pipeline_port_stride"],
    )

    # RISC-V host CPU and its operator workload.
    builder.add_cpu(cpu_id=0)
    builder.set_workload(os.path.abspath(binary), argv=workload_args)

    # NPU command queue and the matrix processing unit.
    builder.add_megacmdqueue()
    builder.add_mpu(
        attr_name="mpu",
        device_id=0,
        num_mem_side_ports=mpu["mem_side_ports"],
        mem_port_outstanding_limit=mpu["outstanding_limit"],
        array_dim=mpu["array_dim"],
        a_buffer_capacity_bytes=mpu["a_buffer_bytes"],
        b_buffer_capacity_bytes=mpu["b_buffer_bytes"],
        c_buffer_capacity_bytes=mpu["c_buffer_bytes"],
    )

    # Vector engine and its nonlinear lookup-table pipeline.
    lut = builder.add_lut(
        range_reduction_latency=lut_config["range_reduction_latency"],
        lookup_latency=lut_config["lookup_latency"],
        interpolation_latency=lut_config["interpolation_latency"],
        normalize_latency=lut_config["normalize_latency"],
        dlen_bytes=vector["dlen_bytes"],
    )
    builder.add_vpu(
        vpu_id=0,
        num_mem_side_ports=vector["mem_side_ports"],
        mem_port_outstanding_limit=vector["outstanding_limit"],
        input_buffer_count=vector["input_buffer_count"],
        output_buffer_count=vector["output_buffer_count"],
        local_buffer_stride=vector["local_buffer_stride"],
        dlen_bytes=vector["dlen_bytes"],
        int32_cycles_per_dlen=vector["int32_cycles_per_dlen"],
        float16_cycles_per_dlen=vector["float16_cycles_per_dlen"],
        float32_cycles_per_dlen=vector["float32_cycles_per_dlen"],
        lut=lut,
    )

    return builder.instantiate_root()


def map_workload(top, config):
    process = top.system.cpu.workload[0]
    spm = config["memory"]["spm"]
    dram = config["memory"]["dram"]
    cmd_width_bytes = int(top.system.cmdq.mega_cmd_width) // 8
    process.map(0x70000000, 0x70000000, 2 * cmd_width_bytes, False)
    process.map(0x71000000, 0x71000000, 4, False)
    process.map(
        spm["base_address"],
        spm["base_address"],
        spm["size_bytes"],
        False,
    )
    process.map(0x20000000, 0x20000000, dram["size_bytes"], False)
