# FlashAttention cycle simulation

Run the default F7 acceptance case from the host:

```sh
./fproject/sim/test-f7.sh
```

Run a specific task:

```sh
./fproject/sim/gem5-fa-sim.sh \
  --hardware fproject/ppa_eval/configs/f7.yaml \
  --task fproject/sim/configs/fa_q128_kv512_d128.yaml
```

The shell entry runs gem5 in `${USER}.gem5`. PPA and cycle simulation both use
the 2 GHz hardware clock. The 64×64 MPU provides 4096 MAC/cycle, or 8192
operations/cycle when multiply-add is counted as two operations, for a
16.384 TOPS peak. Performance uses summed NPU profile busy cycles; elapsed
cycles are retained as a diagnostic.

For the default q=256, kv=1024, d=128 task, the calibrated model performs
134,217,728 tensor operations in 35,440 busy cycles. This is 7.574 TOPS,
or 46.23% of the 16.384 TOPS physical peak; `test-f7.sh` accepts 40–55%.
The raw profile separates 17,792 MPU busy cycles (16,384 compute cycles)
from 17,648 VPU softmax busy cycles.

Task files explicitly configure `tiling.q=128` and `tiling.kv=128`, mapped to
FlashAttention `BR=BC=128`. QK, online softmax, and PV are all invoked once per
query/KV tile; the summary records block counts and validates profile macro
counts derived from the runtime shape. Each 128×128 attention tile is
internally executed on the configured 64×64 physical MPU. A partial final
query tile is padded to
BR=128 for the softmax primitive's fixed layout unit; the summary reports
`padded_q_rows`, while QK/PV retain the logical row count.

The hardware YAML intentionally carries two different SRAM/SPM capacities:

- `memory.sram.capacity_bytes=262144` is the 256 KiB logical capacity used by
  PPA evaluation. Its `port_groups` additionally provision enough 4096×128
  macros for 2R1W bandwidth, so the reported physical macro capacity is larger.
- `simulation.spm.size_bytes=8388608` is a sufficiently large functional
  workspace used by cycle simulation.

The cycle simulator does not use the 8 MiB workspace as PPA capacity.

Each run creates an independent directory under `results/` containing the
input snapshots, resolved simulation configuration, gem5 stdout/stderr, raw
profile log, parsed JSON, interactive HTML, manifest, and performance summary.
