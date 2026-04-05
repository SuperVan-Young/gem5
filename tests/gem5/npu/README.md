# NPU Test Suite Overview

This directory contains gem5 integration tests for the NPU-related model under
`src/npu/mega/`. The tests are not random unit fragments; together they define
the current executable contract for:

- macro-command MMIO helpers
- `MegaCmdQueue`
- sync indicator behavior
- generic `SpecializedExecutionUnit`
- multiport SEU behavior against SPM
- `ScratchpadMemory`
- `VpuUnit`
- `LutUnit`
- `DmaUnit`
- multi-CPU command submission through independent queue input ports

## Test Layout

- `configs/`: shared Python builder used by many tests.
- `utils/`: reusable C/C++ test helpers for MMIO, command packing, and sync.
- `mega/`: command-queue-only tests.
- `seu/`: base SEU tests without SPM data movement.
- `seu_multiport/`: SEU multiport read/write/repetition tests with SPM.
- `spm/`: scratchpad memory tests.
- `vpu/`: legacy multi-instance VPU tests.
- `vpu_decode_smoke/`: command decode and opcode-shape smoke tests.
- `vpu_elemwise/`: `VADD/VSUB/VMUL/VDIV` coverage.
- `vpu_unary/`: `VSCALE/VCVT/VSQRT/VEXP` coverage.
- `vpu_fma/`: `VFMA` coverage.
- `vpu_reduce/`: `VREDUCE_SUM/VREDUCE_MAX` coverage.
- `vpu_loadstore/`: `VLOAD/VSTORE` coverage.
- `vpu_softmax/`: `VSOFTMAX` correctness plus LUT timing/accounting checks.
- `dma/`: DMA tensor transfer tests.
- `system_basic/`: minimal integrated control path with one linear op plus one
  LUT op.
- `system_vpu_dual/`: single-CPU test for `2x VPU + shared LUT`.
- `system_pipeline/`: `DMA + linear VPU + Softmax` integrated path.
- `system_multiport/`: 2-CPU shared-control test on top of the shared-LUT
  system pipeline.
- `sit/`: sync-indicator interaction tests.
- `tile/`: end-to-end queue + SEU flow test.
- `4rv/`: 4-core stress test for shared queue infrastructure.

## Shared Infrastructure

### `configs/npu_test_system.py`

`NPUTestSystemBuilder` is the common system-construction helper. It hides most
of the repetitive gem5 wiring:

- builds a timing-mode `System` with a single `SystemXBar`.
- can attach `SimpleMemory`, `ScratchpadMemory`, CPUs, workloads, queue, SEU,
  VPU, and DMA.
- can build a config-level grouped NPU subsystem with `add_mega_seu()`, which
  currently owns `lut`, `vpu0`, `vpu1`, and optional `dma`.
- maps user processes into the NPU MMIO regions.
- provides helper methods for per-port `MegaCmdQueue` mapping.

The builder also makes the current address map explicit:

- command queue base: `0x7000_0000`
- per-port stride: `1 MiB`
- sync MMIO base: `0x7100_0000`
- default SEU/VPU base: `0x7200_0000`
- DMA base: `0x7400_0000`
- SPM base: `0x6000_0000`
- DRAM base used by tests: `0x2000_0000`

Because all devices connect to one `SystemXBar`, the tests also implicitly
exercise the "data and control share one bus" design used by the current model.

Current recommended system-level ownership shape:

- `system.cmdq`
- `system.seu.lut`
- `system.seu.vpu0`
- `system.seu.vpu1`
- optional `system.seu.dma`

This is currently a config-level grouping boundary, not yet a dedicated C++
composite hardware model.

### `utils/`

The helper headers here define the software-visible command contract:

- `npu_mmio.hh`: raw 32-bit MMIO read/write helpers.
- `cmd/common.hh`: `NpuCmd` container, common header bitfields, launch helpers,
  and device-type enum.
- `npu_sync.hh`: sync-wait command creation, sync-set MMIO writes, and
  `sync_done` helper.

Important note:

- `NPU_DEVICE_TYPE_MPU = 0x3` is reserved in helpers, but there is no MPU
  implementation or MPU test yet.

## What Each Test Covers

### `mega/`

Files:

- `test_megacmdqueue.py`
- `configs/megacmdqueue_full.py`
- `src/megacmdqueue_mmio_riscv.c`

Purpose:

- validates raw `MegaCmdQueue` MMIO behavior without attaching downstream sync
  logic.
- the workload pushes two commands, performs an explicit `pop`, then pushes one
  more command.
- the config expects final queue occupancy `== 2`.

This is mainly a queue staging/control-register sanity test.

### `seu/`

Files:

- `test_seu.py`
- `configs/seu_basic.py`
- `src/seu_mmio_riscv.c`

Purpose:

- validates the base `SpecializedExecutionUnit` in its simplest form.
- the workload issues 5 VPU-typed commands directly.
- read mask and write mask are zero, so execution uses no SPM transactions.
- the config checks counters exposed by the SEU:
  - completed commands
  - prologues
  - executes
  - epilogues
  - iterations
  - read/write response counts
  - queue empty and not busy at end

This is the clearest test for the generic SEU phase machine.

### `seu_multiport/`

Files:

- `test_seu_multiport.py`
- `configs/seu_multiport.py`
- `src/seu_multiport_riscv.c`

Purpose:

- validates multiport SEU behavior against real `ScratchpadMemory`.
- uses `num_mem_side_ports = 4`.
- seeds four SPM slots, launches one command with:
  - `read_mask = 0xB` (ports 0, 1, 3)
  - `write_mask = 0xE` (ports 1, 2, 3)
  - `repetition = 3`
- workload computes the expected slot values in software and polls SPM until
  the hardware model converges.
- config also checks the exported SEU counters.

This is the best reference for understanding the SEU/VPU default read-signature
and writeback behavior.

### `spm/`

Files:

- `test_spm.py`
- `configs/spm_basic.py`
- `src/spm_test.c`

Purpose:

- validates basic read/write behavior of `ScratchpadMemory`.
- the Python config is intentionally simple and only checks normal process exit.

If SPM regressions appear, inspect the workload and saved artifacts under
`tests/testing-results/`.

### `vpu/`

Files:

- `test_vpu.py`
- `configs/vpu_basic.py`
- `src/vpu_test_riscv.c`

Purpose:

- validates two separate `VpuUnit` instances (`vpu0`, `vpu1`) sharing the same
  SPM.
- commands target different `device_id` values.
- each VPU receives two commands with different masks and repetitions.
- workload computes expected final SPM slot contents and per-VPU stats.
- config checks both instances independently:
  - completed commands
  - prologues
  - executes
  - epilogues
  - read/write responses
  - iterations

This is the main proof that the current model can represent "two NPU units"
using separate `VpuUnit` instances.

### `vpu_decode_smoke/`, `vpu_elemwise/`, `vpu_unary/`, `vpu_fma/`,
### `vpu_reduce/`, `vpu_loadstore/`, `vpu_softmax/`

These directories are the current per-feature VPU regression set.

Purpose:

- `vpu_decode_smoke/`: quick sanity checks for command encoding and decode
- `vpu_elemwise/`: verifies elementwise arithmetic behavior
- `vpu_unary/`: verifies unary operations and linear-vs-LUT timing separation
- `vpu_fma/`: verifies fused multiply-add behavior
- `vpu_reduce/`: verifies reduction operations
- `vpu_loadstore/`: verifies resident-buffer `VLOAD/VSTORE` behavior
- `vpu_softmax/`: verifies `VSOFTMAX` correctness plus direct and mirrored LUT
  statistics

Together, these tests define the currently supported VPU opcode set more
accurately than the older `vpu/` directory alone.

### `dma/`

Files:

- `test_dma.py`
- `configs/dma_proto.py`
- `src/dma_proto_riscv.c`

Purpose:

- validates `DmaUnit` command decode, address checking, batching, and data
  movement across DRAM and SPM.
- supported scenarios:
  - `basic_dram_to_spm`
  - `basic_spm_to_dram`
  - `hwc_to_blocked`
  - `blocked_to_blocked`
  - `buffer_size_forces_batching`
  - `sync_completion`
  - `invalid_address`

Notable details:

- `buffer_size_forces_batching` shrinks DMA internal buffer size to force
  multi-batch transfer planning.
- `sync_completion` checks DMA completion ordering through sync wait plus a
  second DMA command.
- `invalid_address` is expected to panic gem5; the test harness treats that as
  success and matches the panic text on stderr.

This directory is the authoritative spec for the currently implemented DMA
command format.

### `system_basic/`

Files:

- `test_system_basic.py`
- `configs/system_basic.py`
- `src/system_basic_riscv.c`

Purpose:

- validates the minimum queue-driven path after the LUT split.
- validates the minimal integrated control path:
  - CPU
  - `MegaCmdQueue`
  - sync wait handling
  - one `VpuUnit`
  - one attached `LutUnit`
  - deferred `sync_done` completion
- workload sequence:
  - queue one linear VPU op
  - wait for its sync completion
  - queue one LUT-backed unary op
  - call `npu_cmd_sync_done()`
- config checks:
  - normal process exit
  - drained command queue
  - drained VPU queue
  - LUT request and command counters
  - LUT completion later than linear completion
  - VPU counters for completed commands, phases, iterations, and responses

This is the recommended first bring-up test for the queue-driven nonlinear
control path.

### `system_vpu_dual/`

Files:

- `test_system_vpu_dual.py`
- `configs/system_vpu_dual.py`
- `src/system_vpu_dual_riscv.c`

Purpose:

- validates a single-CPU scenario with:
  - one `MegaCmdQueue`
  - two `VpuUnit` instances
  - one shared `LutUnit`
  - shared `ScratchpadMemory`
- workload launches:
  - one linear op on `vpu0`
  - one LUT-backed op on `vpu1`
  - explicit sync release and final `npu_cmd_sync_done()`
- config checks:
  - normal process exit
  - drained command queue
  - drained VPU queues
  - shared LUT request and command counters
  - shared LUT completion tick matches the LUT-using VPU
  - LUT-backed VPU completes later than the linear VPU
  - exact per-VPU counters for completed commands, phases, iterations, and
    responses

This is the recommended bridge between `system_basic/` and the DMA-backed
pipeline tests.

### `system_pipeline/`

Files:

- `test_system_pipeline.py`
- `configs/system_pipeline.py`
- `src/system_pipeline_riscv.c`

Purpose:

- validates a chained near-real pipeline with:
  - one `MegaCmdQueue`
  - one `DmaUnit`
  - one shared `LutUnit`
  - two `VpuUnit` instances
  - shared `ScratchpadMemory`
  - DRAM-backed `SimpleMemory`
- workload sequence:
  - seed DRAM with four 32-bit slot values
  - DMA copies them into SPM
  - sync wait on DMA completion
  - VPU0 performs one linear preprocessing op
  - sync wait on VPU0 completion
  - VPU1 performs `VSOFTMAX`
  - sync wait on VPU1 completion
  - for `copy_back`, a second DMA copies the final softmax vector back to DRAM
  - final sync wait plus `npu_cmd_sync_done()`
- workload computes the final linear and softmax contents in software.
- `spm_only` validates the SPM-visible path.
- `copy_back` additionally validates the copied-back DRAM words.
- config checks:
  - normal process exit
  - drained queue, DMA, and VPU state
  - DMA command completion count
  - shared LUT request and command counters
  - softmax execute latency later than the linear path
  - exact per-VPU counters

This is the primary reference for the current
"DMA -> linear VPU -> softmax VPU" system path.

### `system_multiport/`

Files:

- `test_system_multiport.py`
- `configs/system_multiport.py`
- `src/system_multiport_riscv.c`

Purpose:

- validates `MegaCmdQueue(num_input_port=2)` with shared downstream DMA/VPU
  devices and one shared LUT.
- CPU roles:
  - CPU0 seeds DRAM, submits DMA plus a linear VPU op on port 0, and verifies
    final SPM + DRAM contents after global drain.
  - CPU1 uses port 1 to submit the LUT-backed softmax stage and copy-back path.
- the workload uses a small DRAM mailbox for software-side coordination, while
  hardware ordering is still enforced by queue order and sync waits.
- config checks:
  - normal process exit
  - drained command queue, DMA, and VPU state
  - exact DMA completion count
  - shared LUT request and command counters
  - softmax completion later than linear completion
  - exact per-VPU counters

This is the current "2 CPU, shared queue, shared LUT" reference test.

### `sit/`

Files:

- `test_sit_sync_indicator.py`
- `test_sit_launch_sync_launch_sync.py`
- matching config and source files

Purpose:

- validates sync-indicator-table semantics.
- `sit_sync_indicator` launches a sync wait, later sets the indicator through
  MMIO, then fences with `npu_cmd_sync_done()`.
- `sit_launch_sync_launch_sync` checks repeated "launch -> sync wait" behavior.

These tests exercise the `MegaCmdQueue` special handling for
`device_type = sync-indicator-table`.

### `tile/`

Files:

- `test_tile_megacmdqueue_seu.py`
- `configs/tile_megacmdqueue_seu.py`
- `src/tile_megacmdqueue_seu_riscv.c`

Purpose:

- validates the common deployment shape of `CPU -> MegaCmdQueue -> SEU`.
- the workload emits 6 commands, then idles.
- config checks that queue occupancy returns to zero, the SEU queue drains, and
  the SEU completes at least the expected number of commands.

This is the simplest end-to-end pipeline test.

### `4rv/`

Files:

- `test_4rv_sync_stress.py`
- `configs/4rv_sync_stress.py`
- `src/4rv_sync_stress_riscv.c`

Purpose:

- stress-tests shared `MegaCmdQueue` infrastructure with 4 RISC-V CPUs.
- each CPU uses its own queue input port base:
  `0x7000_0000 + cpu_id * 1MiB`.
- each CPU repeatedly enqueues a sync wait and a VPU-like command, then signals
  its sync indicator.
- final check requires:
  - queue empty
  - SEU queue empty
  - SEU completed command count equals `num_cpus * rounds`

This is the strongest concurrency test in the current suite.

## Gaps Visible From The Tests

- There is no MPU-specific test because there is no MPU implementation yet.
- There is still no dedicated C++ composite object that bundles two VPUs, one
  MPU, one DMA, and one LUT behind a single parent SimObject; current tests use
  the config-side `add_mega_seu()` grouping.
- Shared-LUT pressure is covered only at the "single command per VPU" level;
  there is not yet a dense contention regression.
- Most tests validate counters, queue state, sync behavior, and address routing;
  only DMA, SPM, and the newer VPU functional directories validate real payload
  movement.

## Fast Read Order For Future Agents

To understand the test suite quickly, read in this order:

1. `utils/README.md`
2. `utils/cmd/common.hh`
3. `utils/npu_sync.hh`
4. `configs/npu_test_system.py`
5. `vpu_unary/src/vpu_unary_riscv.c`
6. `vpu_softmax/src/vpu_softmax_riscv.c`
7. `system_pipeline/src/system_pipeline_riscv.c`
8. `dma/src/dma_proto_riscv.c`

That sequence gives the command format first, then system wiring, then the most
representative workloads.
