# Copyright (c) 2026
# All rights reserved.

import os
from dataclasses import dataclass

from m5.objects import (
    AddrRange,
    MegaCmdQueue,
    Process,
    RiscvTimingSimpleCPU,
    Root,
    ScratchpadMemory,
    SEWorkload,
    SimpleMemory,
    SpecializedExecutionUnit,
    SrcClockDomain,
    SubSystem,
    System,
    SystemXBar,
    VoltageDomain,
)

try:
    from m5.objects import VpuUnit
except ImportError:
    VpuUnit = None

try:
    from m5.objects import LutUnit
except ImportError:
    LutUnit = None

try:
    from m5.objects import DmaUnit
except ImportError:
    DmaUnit = None

DEFAULT_MEGA_CMD_WIDTH_BITS = 512
DEFAULT_MACRO_CMD_BYTES = DEFAULT_MEGA_CMD_WIDTH_BITS // 8
DEFAULT_CMD_QUEUE_DEPTH = 8
DEFAULT_NUM_SYNC_INDICATOR = 256
DEFAULT_SPM_SIZE = 64 * 1024


@dataclass(frozen=True)
class NPUAddressMap:
    cmdq_base: int = 0x70000000
    sync_base: int = 0x71000000
    seu_base: int = 0x72000000
    cmdq_range_base: int = 0x73000000
    dma_base: int = 0x74000000
    dma_range_base: int = 0x75000000
    spm_base: int = 0x60000000
    dram_base: int = 0x20000000
    cmdq_port_stride: int = 1 << 20
    sync_size: int = 4


class NPUTestSystemBuilder:
    def __init__(
        self,
        clock="1GHz",
        mem_mode="timing",
        mem_ranges=None,
        addr_map=None,
    ):
        self.clock = clock
        self.mem_mode = mem_mode
        self.mem_ranges = mem_ranges or [AddrRange("512MiB")]
        self.addr_map = addr_map or NPUAddressMap()

        self.system = None
        self.root = None
        self.cpus = []
        self.processes = []
        self.components = {}

    def _attach_component(self, component, attr_name, parent=None, keys=None):
        self._require_system()
        owner = self.system if parent is None else parent
        setattr(owner, attr_name, component)
        if keys is None:
            keys = [attr_name]
        for key in keys:
            self.components[key] = component
        return component

    def build_base_system(self):
        self.system = System(
            mem_mode=self.mem_mode,
            mem_ranges=self.mem_ranges,
            membus=SystemXBar(),
            clk_domain=SrcClockDomain(
                clock=self.clock, voltage_domain=VoltageDomain()
            ),
        )
        self.system.system_port = self.system.membus.cpu_side_ports
        return self.system

    def add_default_physmem(self, mem_range=None, attr_name="physmem"):
        self._require_system()
        mem_range = mem_range or self.system.mem_ranges[0]
        memory = SimpleMemory(range=mem_range)
        memory.port = self.system.membus.mem_side_ports
        setattr(self.system, attr_name, memory)
        self.components[attr_name] = memory
        return memory

    def add_lowmem(self, mem_range, attr_name="lowmem"):
        self._require_system()
        memory = SimpleMemory(range=mem_range)
        memory.port = self.system.membus.mem_side_ports
        setattr(self.system, attr_name, memory)
        self.components[attr_name] = memory
        return memory

    def add_spm(
        self,
        base_addr=None,
        size=DEFAULT_SPM_SIZE,
        latency="10ns",
        bandwidth="100GiB/s",
        attr_name="spm",
    ):
        self._require_system()
        base_addr = self.addr_map.spm_base if base_addr is None else base_addr
        spm = ScratchpadMemory(
            range=AddrRange(base_addr, size=size),
            latency=latency,
            bandwidth=bandwidth,
        )
        spm.port = self.system.membus.mem_side_ports
        setattr(self.system, attr_name, spm)
        self.components[attr_name] = spm
        return spm

    def add_cpu(self, cpu_id=0):
        self._require_system()
        cpu = RiscvTimingSimpleCPU(cpu_id=cpu_id)
        cpu.icache_port = self.system.membus.cpu_side_ports
        cpu.dcache_port = self.system.membus.cpu_side_ports
        cpu.createInterruptController()
        self.cpus.append(cpu)
        if cpu_id == 0 and not hasattr(self.system, "cpu"):
            self.system.cpu = cpu
        else:
            setattr(self.system, f"cpu{cpu_id}", cpu)
        return cpu

    def add_cpus(self, num_cpus):
        return [self.add_cpu(cpu_id=i) for i in range(num_cpus)]

    def set_workload(self, binary, argv=None, cpu_id=0, pid=None):
        self._require_cpus()
        binary = os.path.abspath(binary)
        self.system.workload = SEWorkload.init_compatible(binary)
        process = Process(executable=binary)
        if argv is None:
            process.cmd = [binary]
        else:
            process.cmd = [binary, *[str(arg) for arg in argv]]
        if pid is not None:
            process.pid = pid
        cpu = self._get_cpu(cpu_id)
        cpu.workload = process
        cpu.createThreads()
        self._store_process(cpu_id, process)
        return process

    def set_workloads(self, binary, per_cpu_args_fn, pid_base=None):
        binary = os.path.abspath(binary)
        self.system.workload = SEWorkload.init_compatible(binary)
        for cpu_id, cpu in enumerate(self.cpus):
            process = Process(executable=binary)
            args = per_cpu_args_fn(cpu_id)
            process.cmd = [binary, *[str(arg) for arg in args]]
            if pid_base is not None:
                process.pid = pid_base + cpu_id
            cpu.workload = process
            cpu.createThreads()
            self._store_process(cpu_id, process)
        return self.processes

    def add_megacmdqueue(
        self,
        num_input_port=1,
        mega_cmd_width=DEFAULT_MEGA_CMD_WIDTH_BITS,
        cmd_queue_depth=DEFAULT_CMD_QUEUE_DEPTH,
        base_addr=None,
        range_addr=None,
        num_sync_indicator=DEFAULT_NUM_SYNC_INDICATOR,
        attr_name="cmdq",
        parent=None,
        component_keys=None,
    ):
        self._require_system()
        if base_addr is None:
            base_addr = self.addr_map.cmdq_base
        if range_addr is None and num_sync_indicator is not None:
            range_addr = self.addr_map.cmdq_range_base

        kwargs = {
            "num_input_port": num_input_port,
            "mega_cmd_width": mega_cmd_width,
            "cmd_queue_depth": cmd_queue_depth,
            "base_addr": base_addr,
        }
        if range_addr is not None:
            kwargs["range_addr"] = range_addr
        if num_sync_indicator is not None:
            kwargs["num_sync_indicator"] = num_sync_indicator

        cmdq = MegaCmdQueue(**kwargs)
        for _ in range(num_input_port):
            cmdq.cpu_side = self.system.membus.mem_side_ports
        if num_sync_indicator is not None:
            cmdq.sync_indicator_side = self.system.membus.mem_side_ports
            cmdq.mem_side = self.system.membus.cpu_side_ports
        return self._attach_component(
            cmdq,
            attr_name,
            parent=parent,
            keys=component_keys,
        )

    def add_seu(
        self,
        macro_cmd_bytes=DEFAULT_MACRO_CMD_BYTES,
        cmd_queue_depth=DEFAULT_CMD_QUEUE_DEPTH,
        num_mem_side_ports=1,
        base_addr=None,
        debug_process_latency="50ns",
        sync_enqueue_on_data_write=True,
        attr_name="seu",
        parent=None,
        component_keys=None,
    ):
        self._require_system()
        if base_addr is None:
            base_addr = self.addr_map.seu_base
        seu = SpecializedExecutionUnit(
            base_addr=base_addr,
            macro_cmd_bytes=macro_cmd_bytes,
            cmd_queue_depth=cmd_queue_depth,
            num_mem_side_ports=num_mem_side_ports,
            debug_process_latency=debug_process_latency,
            sync_enqueue_on_data_write=sync_enqueue_on_data_write,
        )
        seu.cpu_side = self.system.membus.mem_side_ports
        for _ in range(num_mem_side_ports):
            seu.mem_side = self.system.membus.cpu_side_ports
        return self._attach_component(
            seu,
            attr_name,
            parent=parent,
            keys=component_keys,
        )

    def add_vpu(
        self,
        vpu_id,
        macro_cmd_bytes=DEFAULT_MACRO_CMD_BYTES,
        cmd_queue_depth=DEFAULT_CMD_QUEUE_DEPTH,
        num_mem_side_ports=1,
        num_input_ports=None,
        num_output_ports=None,
        input_buffer_count=2,
        output_buffer_count=2,
        base_addr=None,
        debug_process_latency="50ns",
        sync_enqueue_on_data_write=True,
        lut=None,
        attr_name=None,
        parent=None,
        component_keys=None,
    ):
        self._require_system()
        if VpuUnit is None:
            raise RuntimeError(
                "VpuUnit is not available in the current build"
            )
        if base_addr is None:
            base_addr = self.addr_map.seu_base + (vpu_id << 20)
        if attr_name is None:
            attr_name = f"vpu{vpu_id}"
        kwargs = {
            "device_id": vpu_id,
            "base_addr": base_addr,
            "macro_cmd_bytes": macro_cmd_bytes,
            "cmd_queue_depth": cmd_queue_depth,
            "num_mem_side_ports": num_mem_side_ports,
            "num_input_ports": (
                num_mem_side_ports
                if num_input_ports is None
                else num_input_ports
            ),
            "num_output_ports": (
                num_mem_side_ports
                if num_output_ports is None
                else num_output_ports
            ),
            "input_buffer_count": input_buffer_count,
            "output_buffer_count": output_buffer_count,
            "debug_process_latency": debug_process_latency,
            "sync_enqueue_on_data_write": sync_enqueue_on_data_write,
        }
        if lut is not None:
            kwargs["lut"] = lut
        vpu = VpuUnit(
            **kwargs,
        )
        vpu.cpu_side = self.system.membus.mem_side_ports
        for _ in range(num_mem_side_ports):
            vpu.mem_side = self.system.membus.cpu_side_ports
        return self._attach_component(
            vpu,
            attr_name,
            parent=parent,
            keys=component_keys,
        )

    def add_lut(
        self,
        range_reduction_latency="20ns",
        lookup_latency="30ns",
        interpolation_latency="20ns",
        normalize_latency="20ns",
        table_entries=257,
        attr_name="lut",
        parent=None,
        component_keys=None,
    ):
        self._require_system()
        if LutUnit is None:
            raise RuntimeError(
                "LutUnit is not available in the current build"
            )
        lut = LutUnit(
            range_reduction_latency=range_reduction_latency,
            lookup_latency=lookup_latency,
            interpolation_latency=interpolation_latency,
            normalize_latency=normalize_latency,
            table_entries=table_entries,
        )
        return self._attach_component(
            lut,
            attr_name,
            parent=parent,
            keys=component_keys,
        )

    def add_dma(
        self,
        macro_cmd_bytes=DEFAULT_MACRO_CMD_BYTES,
        cmd_queue_depth=DEFAULT_CMD_QUEUE_DEPTH,
        bank_size=4096,
        base_addr=None,
        sync_enqueue_on_data_write=True,
        attr_name="dma",
        parent=None,
        component_keys=None,
    ):
        self._require_system()
        if DmaUnit is None:
            raise RuntimeError("DmaUnit is not available in the current build")
        if base_addr is None:
            base_addr = self.addr_map.dma_base
        dma = DmaUnit(
            base_addr=base_addr,
            macro_cmd_bytes=macro_cmd_bytes,
            cmd_queue_depth=cmd_queue_depth,
            sync_enqueue_on_data_write=sync_enqueue_on_data_write,
            bank_size=bank_size,
        )
        dma.cpu_side = self.system.membus.mem_side_ports
        dma.mem_side = self.system.membus.cpu_side_ports
        return self._attach_component(
            dma,
            attr_name,
            parent=parent,
            keys=component_keys,
        )

    def add_mega_seu(
        self,
        num_vpus=2,
        include_dma=True,
        macro_cmd_bytes=DEFAULT_MACRO_CMD_BYTES,
        cmd_queue_depth=DEFAULT_CMD_QUEUE_DEPTH,
        vpu_num_mem_side_ports=1,
        vpu_debug_process_latency="50ns",
        vpu_sync_enqueue_on_data_write=True,
        dma_buffer_size=4096,
        dma_sync_enqueue_on_data_write=True,
        lut_range_reduction_latency="20ns",
        lut_lookup_latency="30ns",
        lut_interpolation_latency="20ns",
        lut_normalize_latency="20ns",
        lut_table_entries=257,
        base_addr=None,
        dma_base=None,
        attr_name="seu",
        expose_legacy_component_aliases=True,
    ):
        self._require_system()
        if base_addr is None:
            base_addr = self.addr_map.seu_base

        seu = self._attach_component(SubSystem(), attr_name, keys=[attr_name])

        lut_keys = [f"{attr_name}.lut"]
        if expose_legacy_component_aliases:
            lut_keys.append("lut")
        lut = self.add_lut(
            range_reduction_latency=lut_range_reduction_latency,
            lookup_latency=lut_lookup_latency,
            interpolation_latency=lut_interpolation_latency,
            normalize_latency=lut_normalize_latency,
            table_entries=lut_table_entries,
            attr_name="lut",
            parent=seu,
            component_keys=lut_keys,
        )

        if include_dma:
            dma_keys = [f"{attr_name}.dma"]
            if expose_legacy_component_aliases:
                dma_keys.append("dma")
            self.add_dma(
                macro_cmd_bytes=macro_cmd_bytes,
                cmd_queue_depth=cmd_queue_depth,
                bank_size=dma_buffer_size,
                base_addr=dma_base,
                sync_enqueue_on_data_write=dma_sync_enqueue_on_data_write,
                attr_name="dma",
                parent=seu,
                component_keys=dma_keys,
            )

        for vpu_id in range(num_vpus):
            vpu_keys = [f"{attr_name}.vpu{vpu_id}"]
            if expose_legacy_component_aliases:
                vpu_keys.append(f"vpu{vpu_id}")
            self.add_vpu(
                vpu_id=vpu_id,
                macro_cmd_bytes=macro_cmd_bytes,
                cmd_queue_depth=cmd_queue_depth,
                num_mem_side_ports=vpu_num_mem_side_ports,
                base_addr=base_addr + (vpu_id << 20),
                debug_process_latency=vpu_debug_process_latency,
                sync_enqueue_on_data_write=vpu_sync_enqueue_on_data_write,
                lut=lut,
                attr_name=f"vpu{vpu_id}",
                parent=seu,
                component_keys=vpu_keys,
            )

        return seu

    def instantiate_root(self, full_system=False):
        self._require_system()
        self.root = Root(full_system=full_system, system=self.system)
        return self.root

    def get_process(self, cpu_id=0):
        return self.processes[cpu_id]

    def get_processes(self):
        return list(self.processes)

    def get_cmdq_port_base(self, cpu_id):
        return (
            self.addr_map.cmdq_base + cpu_id * self.addr_map.cmdq_port_stride
        )

    def map_cmdq(self, process=None, cpu_id=0, size=None, base_addr=None):
        process = self._resolve_process(process, cpu_id)
        cmdq = self.components["cmdq"]
        width_bytes = int(cmdq.mega_cmd_width) // 8
        size = 2 * width_bytes if size is None else size
        base_addr = self.addr_map.cmdq_base if base_addr is None else base_addr
        process.map(base_addr, base_addr, size, False)
        return process

    def map_cmdq_port(self, cpu_id, process=None, size=None):
        base_addr = self.get_cmdq_port_base(cpu_id)
        return self.map_cmdq(
            process=process,
            cpu_id=cpu_id,
            size=size,
            base_addr=base_addr,
        )

    def map_seu(self, process=None, cpu_id=0, size=None, base_addr=None):
        process = self._resolve_process(process, cpu_id)
        seu = self.components["seu"]
        macro_cmd_bytes = int(seu.macro_cmd_bytes)
        if size is None:
            size = 2 * macro_cmd_bytes
        base_addr = self.addr_map.seu_base if base_addr is None else base_addr
        process.map(base_addr, base_addr, size, False)
        return process

    def map_vpu(
        self,
        vpu_id,
        process=None,
        cpu_id=0,
        size=None,
        base_addr=None,
        attr_name=None,
    ):
        process = self._resolve_process(process, cpu_id)
        if attr_name is None:
            attr_name = f"vpu{vpu_id}"
        vpu = self.components[attr_name]
        macro_cmd_bytes = int(vpu.macro_cmd_bytes)
        if size is None:
            size = 2 * macro_cmd_bytes
        if base_addr is None:
            base_addr = self.addr_map.seu_base + (vpu_id << 20)
        process.map(base_addr, base_addr, size, False)
        return process

    def map_sync(self, process=None, cpu_id=0, size=None, base_addr=None):
        process = self._resolve_process(process, cpu_id)
        size = self.addr_map.sync_size if size is None else size
        base_addr = self.addr_map.sync_base if base_addr is None else base_addr
        process.map(base_addr, base_addr, size, False)
        return process

    def map_spm(self, process=None, cpu_id=0, base_addr=None, size=None):
        process = self._resolve_process(process, cpu_id)
        spm = self.components["spm"]
        spm_range = spm.range
        base_addr = int(spm_range.start) if base_addr is None else base_addr
        size = int(spm_range.size()) if size is None else size
        process.map(base_addr, base_addr, size, False)
        return process

    def map_dram(self, process=None, cpu_id=0, base_addr=None, size=64 * 1024):
        process = self._resolve_process(process, cpu_id)
        base_addr = self.addr_map.dram_base if base_addr is None else base_addr
        process.map(base_addr, base_addr, size, False)
        return process

    def map_all_required_regions(self):
        for cpu_id, process in enumerate(self.processes):
            if "cmdq" in self.components:
                if len(self.processes) == 1:
                    self.map_cmdq(process=process, cpu_id=cpu_id)
                else:
                    self.map_cmdq_port(cpu_id, process=process)
            if "spm" in self.components:
                self.map_spm(process=process, cpu_id=cpu_id)
        return self.processes

    def _require_system(self):
        if self.system is None:
            raise RuntimeError(
                "Call build_base_system() before adding components."
            )

    def _require_cpus(self):
        self._require_system()
        if not self.cpus:
            raise RuntimeError(
                "Add at least one CPU before assigning workloads."
            )

    def _get_cpu(self, cpu_id):
        try:
            return self.cpus[cpu_id]
        except IndexError as exc:
            raise RuntimeError(f"CPU {cpu_id} has not been created.") from exc

    def _store_process(self, cpu_id, process):
        while len(self.processes) <= cpu_id:
            self.processes.append(None)
        self.processes[cpu_id] = process

    def _resolve_process(self, process, cpu_id):
        if process is not None:
            return process
        try:
            resolved = self.processes[cpu_id]
        except IndexError as exc:
            raise RuntimeError(
                f"Process for CPU {cpu_id} has not been created."
            ) from exc
        if resolved is None:
            raise RuntimeError(
                f"Process for CPU {cpu_id} has not been created."
            )
        return resolved
