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

The hardware config also carries an analytical buffer-pipeline model derived
from the measured MPU and VPU busy cycles:

- The 32-bank baseline exposes 4 B/cycle per bank and one shared 32-bank
  bank set. Tensor and Vector execution therefore remain serialized.
- The 64-bank configuration splits the address space into two independent
  32-bank contexts along the BR/query-block dimension. BC is not used for the
  split because online softmax and output accumulation carry dependencies
  between KV blocks of the same query block.
- The Vector-2x configuration uses 96 banks: 32 banks provide the Tensor
  engine's 128 B/cycle path and 64 banks provide a 256 B/cycle path to two
  parallel Vector groups. It retains the two BR-addressed buffer contexts.

For the default task, the 32-bank model retains 35,440 cycles. The 64-bank
double-buffer model divides QK, softmax, and PV busy time across the two BR
slots and evaluates four finite-pipeline stages:

```text
QK0 -> max(QK1, softmax0) -> max(softmax1, PV0) -> PV1
```

The stages take 4,448, 8,824, 8,824, and 4,448 cycles, respectively, for a
total of 26,544 cycles and a 1.335x speedup. This is a busy-cycle overlap
model; it does not claim that the current sequential workload already issues
the two engines concurrently.

The current workload has a fixed 128-FP32 (512 B) lowest VPU layout
dimension, so the Vector-2x configuration keeps the executable VPU dlen at
512 B. Its analytical model uses `vector_compute_scale=2` to divide the
measured softmax busy cycles between two parallel Vector groups. A full run
measures 8,608 QK cycles, 17,360 softmax cycles, and 8,608 PV cycles. After
Vector scaling, the four finite-pipeline stages total 17,288 cycles, or
15.527 TOPS at 2 GHz. This is 1.535x the modeled throughput of the 64-bank
configuration.

Each run creates an independent directory under `results/` containing the
input snapshots, resolved simulation configuration, gem5 stdout/stderr, raw
profile log, parsed JSON, interactive HTML, manifest, and performance summary.
