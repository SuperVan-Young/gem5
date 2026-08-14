# Atlas-facing performance simulation

This directory contains the small, operator-oriented gem5 interface used by
Atlas.  It is intentionally independent of the FlashAttention orchestration in
`fproject/sim`.

Convert an ATLAS chip description into a gem5 hardware configuration with:

```sh
./fproject-atlas/hw/config_adaptor.py \
  /home/xuechenhao/ATLAS-MICRO-2026/fproject/configs/attention_chip.yaml \
  fproject-atlas/configs/attention_chip.yaml
```

The adaptor scales the MPU, VPU, local buffers, scratchpad, and interconnect
from `configs/gem5_template.yaml`. It resolves ATLAS's referenced HBDRAM YAML
to derive capacity, bandwidth, and closed-row read latency. LUT parameters
remain those of the template.

`memory.dram.capacity_bytes` and `mapped_size_bytes` are the per-core capacity
decoded from the ATLAS HBDRAM organization. DRAM bandwidth is likewise divided
by `core_num`, because one gem5 instance models one ATLAS core.

The first supported operator is int8 matrix multiplication. Run the default
shape from the host with:

```sh
./fproject-atlas/sim/matmul/run.sh \
  --hardware fproject-atlas/configs/attention_chip.yaml \
  --m 256 --n 1024 --k 128
```

The command builds the workload when needed, runs gem5 in `${USER}.gem5`, and
prints the path to `result.json`. It also generates `npu_profile.log`, parsed
`npu_profile.json`, and an interactive `npu_profile.html`. The result compares:

- `naive_cycles`: `ceil(M * N * K / mac_capacity)`;
- `gem5_compute_cycles`: the MPU's tiled compute latency;
- `gem5_busy_cycles`: total MPU busy time, including data movement and command
  pipeline overhead.

`gem5_busy_cycles` is read directly from the MPU model's `busyCycles()`
statistic. The profiling summary independently converts the profiled macro
tick span to cycles, making the two measurements easy to cross-check.

`hw/top.py` is the hardware composition entry point. It reads one hardware
YAML and builds the base system, memory, CPU, command queue, and MPU in one
annotated sequence before returning an m5 `Root` instance.
