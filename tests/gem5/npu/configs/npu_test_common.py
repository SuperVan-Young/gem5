# Copyright (c) 2026
# All rights reserved.

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NPUTestBuildContext:
    builder: Any
    binary: str
    processes: list[Any]
    root: Any = None

    @property
    def system(self):
        return self.builder.system

    def get_process(self, cpu_id=0):
        return self.processes[cpu_id]

    def get_component(self, name):
        return self.builder.components[name]


def make_builder(**builder_kwargs):
    from npu_test_system import NPUTestSystemBuilder

    return NPUTestSystemBuilder(**builder_kwargs)


def _normalize_argv(argv):
    if argv is None:
        return None
    return [str(arg) for arg in argv]


def _resolve_per_cpu_args(per_cpu_argv, cpu_id):
    if per_cpu_argv is None:
        return []
    if callable(per_cpu_argv):
        args = per_cpu_argv(cpu_id)
    else:
        args = per_cpu_argv[cpu_id]
    return [str(arg) for arg in args]


def build_single_cpu_context(
    binary,
    *,
    argv=None,
    cpu_id=0,
    builder_kwargs=None,
    add_default_physmem=True,
    add_cmdq=False,
    add_seu=False,
    add_spm=False,
    map_cmdq=None,
    map_seu=None,
    map_spm=None,
    map_sync=False,
    map_dram=False,
    cmdq_kwargs=None,
    seu_kwargs=None,
    spm_kwargs=None,
    instantiate_root=True,
):
    builder = make_builder(**(builder_kwargs or {}))
    builder.build_base_system()

    if add_default_physmem:
        builder.add_default_physmem()
    if add_spm:
        builder.add_spm(**(spm_kwargs or {}))

    builder.add_cpu(cpu_id=cpu_id)
    process = builder.set_workload(
        os.path.abspath(binary),
        argv=_normalize_argv(argv),
        cpu_id=cpu_id,
    )

    if add_cmdq:
        builder.add_megacmdqueue(**(cmdq_kwargs or {}))
    if add_seu:
        builder.add_seu(**(seu_kwargs or {}))

    if map_cmdq is None:
        map_cmdq = add_cmdq
    if map_seu is None:
        map_seu = add_seu
    if map_spm is None:
        map_spm = add_spm

    if map_cmdq:
        builder.map_cmdq(process=process, cpu_id=cpu_id)
    if map_seu:
        builder.map_seu(process=process, cpu_id=cpu_id)
    if map_spm:
        builder.map_spm(process=process, cpu_id=cpu_id)
    if map_sync:
        builder.map_sync(process=process, cpu_id=cpu_id)
    if map_dram:
        if not add_default_physmem:
            raise ValueError("map_dram requires add_default_physmem=True")
        builder.map_dram(process=process, cpu_id=cpu_id)

    root = builder.instantiate_root() if instantiate_root else None
    return NPUTestBuildContext(
        builder=builder,
        binary=os.path.abspath(binary),
        processes=builder.get_processes(),
        root=root,
    )


def map_single_cpu_regions(
    builder,
    *,
    process=None,
    cpu_id=0,
    map_cmdq=False,
    map_seu=False,
    map_spm=False,
    map_sync=False,
    map_dram=False,
):
    if process is None:
        process = builder.get_process(cpu_id)

    if map_cmdq:
        builder.map_cmdq(process=process, cpu_id=cpu_id)
    if map_seu:
        builder.map_seu(process=process, cpu_id=cpu_id)
    if map_spm:
        builder.map_spm(process=process, cpu_id=cpu_id)
    if map_sync:
        builder.map_sync(process=process, cpu_id=cpu_id)
    if map_dram:
        builder.map_dram(process=process, cpu_id=cpu_id)


def build_multi_cpu_context(
    binary,
    *,
    cpu_count,
    per_cpu_argv=None,
    pid_base=None,
    builder_kwargs=None,
    add_default_physmem=True,
    add_cmdq=False,
    add_spm=False,
    map_cmdq_ports=True,
    map_spm=False,
    map_sync=False,
    map_dram=False,
    cmdq_kwargs=None,
    spm_kwargs=None,
    instantiate_root=True,
):
    builder = make_builder(**(builder_kwargs or {}))
    builder.build_base_system()

    if add_default_physmem:
        builder.add_default_physmem()
    if add_spm:
        builder.add_spm(**(spm_kwargs or {}))

    builder.add_cpus(cpu_count)
    builder.set_workloads(
        os.path.abspath(binary),
        lambda cpu_id: _resolve_per_cpu_args(per_cpu_argv, cpu_id),
        pid_base=pid_base,
    )

    if add_cmdq:
        kwargs = dict(cmdq_kwargs or {})
        kwargs.setdefault("num_input_port", cpu_count)
        builder.add_megacmdqueue(**kwargs)
    if map_cmdq_ports and not add_cmdq:
        raise ValueError("map_cmdq_ports requires add_cmdq=True")
    if map_dram and not add_default_physmem:
        raise ValueError("map_dram requires add_default_physmem=True")

    processes = builder.get_processes()
    map_multi_cpu_regions(
        builder,
        processes=processes,
        map_cmdq_ports=map_cmdq_ports,
        map_spm=map_spm,
        map_sync=map_sync,
        map_dram=map_dram,
    )

    root = builder.instantiate_root() if instantiate_root else None
    return NPUTestBuildContext(
        builder=builder,
        binary=os.path.abspath(binary),
        processes=processes,
        root=root,
    )


def map_multi_cpu_regions(
    builder,
    *,
    processes=None,
    map_cmdq_ports=False,
    map_spm=False,
    map_sync=False,
    map_dram=False,
):
    if processes is None:
        processes = builder.get_processes()

    for cpu_id, process in enumerate(processes):
        if map_cmdq_ports:
            builder.map_cmdq_port(cpu_id, process=process)
        if map_spm:
            builder.map_spm(process=process, cpu_id=cpu_id)
        if map_sync:
            builder.map_sync(process=process, cpu_id=cpu_id)
        if map_dram:
            builder.map_dram(process=process, cpu_id=cpu_id)


def collect_component_snapshot(component, field_specs):
    snapshot = {}
    for key, accessor in field_specs.items():
        if callable(accessor):
            snapshot[key] = accessor(component)
            continue

        value = getattr(component, accessor)
        snapshot[key] = value() if callable(value) else value
    return snapshot


def collect_builder_component_snapshot(builder, component_name, field_specs):
    try:
        component = builder.components[component_name]
    except KeyError as exc:
        raise KeyError(f"Component '{component_name}' was not added.") from exc

    return collect_component_snapshot(component, field_specs)


def collect_cmdq_snapshot(builder, component_name="cmdq"):
    return collect_builder_component_snapshot(
        builder,
        component_name,
        {"queue_occupancy": "queueOccupancy"},
    )


def collect_seu_snapshot(builder, component_name="seu"):
    return collect_builder_component_snapshot(
        builder,
        component_name,
        {
            "queue_occupancy": "queueOccupancy",
            "issue_busy": "isIssueBusy",
            "completed_cmds": "completedCmdCount",
            "prologues": "prologueCount",
            "executes": "executeCount",
            "epilogues": "epilogueCount",
            "iterations": "completedIterationCount",
            "read_resps": "completedReadRespCount",
            "write_resps": "completedWriteRespCount",
        },
    )


def collect_vpu_snapshot(builder, component_name="vpu0"):
    return collect_builder_component_snapshot(
        builder,
        component_name,
        {
            "queue_occupancy": "queueOccupancy",
            "issue_busy": "isIssueBusy",
            "completed_cmds": "completedCmdCount",
            "prologues": "prologueCount",
            "executes": "executeCount",
            "epilogues": "epilogueCount",
            "iterations": "completedIterationCount",
            "read_resps": "completedReadRespCount",
            "write_resps": "completedWriteRespCount",
        },
    )


def emit_summary(prefix, snapshot):
    prefix = prefix.upper()
    for key, value in snapshot.items():
        print(f"{prefix}_{key.upper()}={value}")


def verify_snapshot(observed, expected):
    return observed == expected
