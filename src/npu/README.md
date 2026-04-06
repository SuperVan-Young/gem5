# NPU Module Overview

This directory contains the NPU-side simulator implementation currently under
`src/npu/`. The code is centered around a command-queue plus execution-unit
model:

- `MegaCmdQueue`: accepts macro commands from one or more CPU MMIO ports,
  buffers them, and dispatches them to target devices by decoding the command
  header.
- `SpecializedExecutionUnit` (SEU): a generic command-consuming execution unit
  with a reusable state machine for `prologue -> mvin -> execute -> mvout ->
  epilogue`.
- `VpuUnit`: a concrete SEU subclass used by current "NPU" tests.
- `LutUnit`: a standalone nonlinear helper resource used by one or more
  `VpuUnit` instances.
- `DmaUnit`: a DMA-oriented command consumer. It subclasses SEU but overrides
  the execution flow with its own batched gather/scatter pipeline.
- `ScratchpadMemory`: timing-mode SRAM/SPM used as accelerator local storage.

## What Is Implemented Today

The current model does not implement a single monolithic "SEU contains two NPU,
one MPU, one DMA" composite object. Instead, the hardware shape is represented
implicitly by command routing plus multiple instantiated devices:

- two NPU-like units are modeled by creating two `VpuUnit` instances with
  different `device_id` values, as in
  `tests/gem5/npu/testcases/vpu/legacy_dual_device_basic/`.
- one DMA is modeled by `DmaUnit`.
- an MPU device type is reserved in the test helper headers
  (`NPU_DEVICE_TYPE_MPU = 0x3`) but there is no MPU SimObject or C++
  implementation in `src/npu` yet.

So the current codebase supports "2x VPU + 1x DMA + shared LUT" concretely,
while "1x MPU" exists only as a command-space reservation.

## Bus And Topology Model

The current gem5 model uses a single `SystemXBar` for both control and data:

- CPUs issue MMIO writes to `MegaCmdQueue`, `SpecializedExecutionUnit`,
  `VpuUnit`, `DmaUnit`, and the sync-indicator MMIO aperture through the same
  system bus.
- `MegaCmdQueue`, `SpecializedExecutionUnit`, `VpuUnit`, and `DmaUnit` also use
  that same bus for memory-side traffic.
- `ScratchpadMemory` is attached to the same bus.

That means "control" traffic (MMIO command submission, sync writes) and "data"
traffic (SPM/DRAM reads and writes) are not separated in the simulator right
now. They share the same interconnect and therefore the same backpressure path.

## Address Map And Command Routing

The test builder in `tests/gem5/npu/configs/npu_test_system.py` is the easiest
place to see the intended address map:

- `0x6000_0000`: scratchpad memory base.
- `0x7000_0000 + port_id * 1MiB`: `MegaCmdQueue` per-input-port MMIO window.
- `0x7100_0000`: sync-indicator MMIO window.
- `0x7200_0000 + device_id * 1MiB`: default VPU/SEU MMIO space used by tests.
- `0x7400_0000`: DMA MMIO base.

`MegaCmdQueue` dispatches queued commands by taking the first 32-bit word:

- `device_type` = bits `[31:28]`
- `device_id` = bits `[27:24]`
- `op_code` = bits `[23:16]`
- `sync_indicator` = bits `[15:8]`
- `set_indicator_sns` = bit `[7]`
- `set_indicator_snd` = bit `[6]`

The dispatch target address is:

- `0x7000_0000 | (device_type << 24) | (device_id << 20)`

Examples:

- `device_type = 0x2`, `device_id = 0` routes to `0x7200_0000` (`VpuUnit 0`).
- `device_type = 0x2`, `device_id = 1` routes to `0x7210_0000` (`VpuUnit 1`).
- `device_type = 0x4`, `device_id = 0` routes to `0x7400_0000` (`DmaUnit`).

`device_type = 0x1` with `op_code = 0` is treated specially by
`MegaCmdQueue` as a sync-wait command and is not forwarded to another device.
It stalls at the queue head until the corresponding sync indicator becomes `1`.

## Core Objects

### `MegaCmdQueue`

Files:

- `MegaCmdQueue.hh`
- `MegaCmdQueue.cc`
- `MegaCmdQueue.py`

Responsibilities:

- exposes `cpu_side[]` vector response ports, one MMIO staging window per input
  port.
- exposes one dedicated `sync_indicator_side` MMIO port at `0x7100_0000`.
- stores one staging buffer per CPU input port.
- supports three control writes to the control word (`base + cmd_bytes`):
  `push=0`, `pop=1`, `sync_done=2`.
- enforces a same-cycle enqueue gate with `hasEnqueuedCmd`, so a port cannot
  push twice in one cycle.
- buffers complete macro commands in an internal FIFO.
- dispatches FIFO head entries to the target device by issuing a memory-side
  write carrying the full command buffer.
- maintains a `syncIndicatorTable`; sync-wait commands block until the indexed
  indicator is set, then the entry is consumed and the command is popped.
- supports deferred completion of `sync_done`: the CPU MMIO response can be held
  until the command queue drains and no write is in flight.

Important details:

- per-port MMIO windows are `2 * megaCmdBytes` large: first half is data
  staging, second half is the control register.
- queue occupancy is visible through the exported `queueOccupancy()` method.
- only timing mode is supported; atomic/functional CPU accesses are rejected.

### `SpecializedExecutionUnit`

Files:

- `SpecializedExecutionUnit.hh`
- `SpecializedExecutionUnit.cc`
- `SpecializedExecutionUnit.py`

This is the generic execution framework for command consumers. A command can be
written directly to the SEU MMIO window or forwarded there by `MegaCmdQueue`.

Execution model:

1. command arrives into the staging buffer and is enqueued.
2. `issueOneCommand()` pops one queued command and starts execution.
3. `beginActiveCommand()` decodes command header and multiport words:
   - word 1: `readMask`
   - word 2: `writeMask`
   - word 3: `repetition`
   - word 4: `reserved`
4. the phase machine advances through:
   - `Prologue`
   - `LaunchingMvin`
   - `WaitingMvin`
   - `Executing`
   - `LaunchingMvout`
   - `WaitingMvout`
   - `Epilogue`
   - `Completing`
5. after command completion, the unit may emit a sync-set write to
   `0x7100_0000` when header bits `set_indicator_sns` or `set_indicator_snd`
   are set.

Default memory behavior:

- if subclasses do not build explicit requests, the base class generates
  default 32-bit reads/writes to SPM slots:
  `0x6000_0000 + port_id * 0x40`.
- read values are collected in `readResults`.
- default writeback computes a synthetic mixed value from the read signature,
  iteration count, and port id.

Extension hooks for subclasses:

- `onCommandBegin`
- `prologue`
- `buildMvinRequests`
- `onMvinResponse`
- `execute`
- `buildMvoutRequests`
- `onMvoutResponse`
- `epilogue`
- `shouldExit`

The class also exports runtime counters used heavily by tests:

- completed commands
- queue occupancy
- busy flag
- read/write response counts
- completed iterations
- prologue/execute/epilogue counts

### `VpuUnit`

Files:

- `VpuUnit.hh`
- `VpuUnit.cc`
- `VpuUnit.py`

`VpuUnit` is the concrete vector-style execution unit currently used in tests.
It still relies on the common SEU phase machine, but its `execute()` path now
implements explicit vector operations instead of the older synthetic signature
writeback model.

#### Current VPU Op Coverage

Implemented opcodes today:

- elementwise arithmetic:
  - `VADD`
  - `VSUB`
  - `VMUL`
  - `VDIV`
- unary / conversion:
  - `VSCALE`
  - `VCVT_I2F`
  - `VCVT_F2I`
  - `VSQRT`
  - `VEXP`
- compound / reduction:
  - `VFMA`
  - `VREDUCE_SUM`
  - `VREDUCE_MAX`
  - `VSOFTMAX`
- local-buffer movement:
  - `VLOAD`
  - `VSTORE`
- legacy path:
  - `op_code = 0` is still accepted as the old compatibility opcode

Supported data types today:

- `Int32`
- `Float32`

Command shape:

- `readMask` and `writeMask` still define which SPM slots participate.
- command words 4-9 carry vector metadata such as element count, source and
  destination stride, data type, and scalar bits.
- `repetition` still comes from the common SEU contract.

Current scope and caveats:

- this is a functional plus timing-shape model, not a calibrated production
  vector ISA timing model.
- only timing mode is supported.
- `VSoftmax` is implemented as one VPU command with internal
  `reduce_max + exp + reduce_sum + normalize` steps, but those substages are
  not yet exported as separate timing counters.

#### Nonlinear Path Through `LutUnit`

`VSQRT`, `VEXP`, and `VSOFTMAX` no longer depend on hidden host-side nonlinear
timing inside `VpuUnit`. They now use an explicit `LutUnit` resource.

Current `LutUnit` model:

- one standalone SimObject
- one shared resource
- fixed per-request latency
- single serialized queue through `availableTick`
- no bank model, queue-depth model, or advanced arbitration yet

Current nonlinear behavior:

- `VSQRT`: reserve LUT resource, then evaluate sqrt approximation
- `VEXP`: reserve LUT resource, then evaluate exp approximation
- `VSOFTMAX`: reserve LUT resource for the exp substep and execute:
  1. `reduce_max`
  2. shifted `exp` through `LutUnit`
  3. `reduce_sum`
  4. normalize and write back

This is the current intended interpretation: the structure and timing boundary
is explicit, but the detailed cycle model is still intentionally simple.

#### Shared LUT Topology

The current codebase does not yet enforce a monolithic "SEU owns 2 VPU + 1 DMA
+ 1 LUT" object graph. Instead, tests and configs instantiate separate
SimObjects and wire them together explicitly.

The supported pattern today is:

- multiple `VpuUnit` instances may point their `lut` parameter at the same
  `LutUnit`
- configs may read either:
  - per-VPU mirror accessors such as `lastLutExecuteLatency()`
  - direct shared-resource counters such as `shared_lut.requestCount()`

That is how current system tests model "two VPU + one shared LUT".

The current recommended config-side entry point for this object graph is
`NPUTestSystemBuilder.add_mega_seu()` in
`tests/gem5/npu/configs/npu_test_system.py`. It creates one `SubSystem`
containing:

- `seu.lut`
- `seu.vpu0`
- `seu.vpu1`
- optional `seu.dma`

This is a config-level ownership boundary today, not yet a dedicated composite
C++ hardware model.

#### Runtime Observability

The VPU/LUT path exports enough state for both functional and timing-focused
tests.

From `VpuUnit`:

- `lutRequestCount()`
- `lutCommandCount()`
- `lastLinearExecuteLatency()`
- `lastLutExecuteLatency()`
- `lastSoftmaxExecuteLatency()`
- `lastLinearCompletionTick()`
- `lastLutCompletionTick()`
- `lastSoftmaxCompletionTick()`

From `LutUnit`:

- `requestCount()`
- `commandCount()`
- `lastExecuteLatency()`
- `lastCompletionTick()`
- `lastSoftmaxExecuteLatency()`
- `lastSoftmaxCompletionTick()`

Recommended interpretation:

- `LutUnit` owns the primary nonlinear-resource statistics
- `VpuUnit` mirror methods exist for compatibility and convenience in
  VPU-centric test configs

### `DmaUnit`

Files:

- `DmaUnit.hh`
- `DmaUnit.cc`
- `DmaUnit.py`

`DmaUnit` is the most specialized component in this directory.

Key properties:

- requires 64-byte commands (`macro_cmd_bytes == 64`).
- validates `device_type == 0x4`.
- decodes tensor transfer metadata from words 3-14.
- supports four transfer modes:
  - DRAM -> SPM
  - SPM -> DRAM
  - SPM -> SPM
  - DRAM -> DRAM
- supports plain HWC layout (`k = 0`) and blocked layout (`k > 0`).
- validates address space membership:
  - DRAM: `0x2000_0000` to `0x5fff_ffff`
  - SPM: `0x6000_0000` to `0x6fff_ffff`
- uses a bounded internal flat buffer (`buffer_size`) and splits large tensor
  copies into batches when needed.
- constructs line-based gather plans for source reads and scatter plans for
  destination updates.
- sends completion sync writes itself and classifies them with its own internal
  `RequestKind`.

The DMA path is more custom than the VPU path:

- it overrides `startExecuteCommand()`.
- it overrides `handleMemResponse()`.
- it does not use the SEU default `mvin/execute/mvout` helpers directly.

### `ScratchpadMemory`

Files:

- `ScratchpadMemory.hh`
- `ScratchpadMemory.cc`
- `ScratchpadMemory.py`

This is a timing-mode SRAM-style memory used as accelerator-local storage.

Behavior:

- inherits from `AbstractMemory`.
- exposes a single response port.
- enforces fixed access latency plus bandwidth-based busy time.
- queues timing responses and supports retry/backpressure.
- allows unconnected initialization for flexibility in test setups.

It is intentionally simple, but it is the backing storage used by multiport SEU
and VPU tests as well as DMA SPM transfers.

## Build Integration

`src/npu/SConscript` registers:

- debug flags: `MegaCmdQueue`, `SpecializedExecutionUnit`,
  `ScratchpadMemory`, `VPU`, `DmaUnit`
- SimObjects: `MegaCmdQueue`, `SpecializedExecutionUnit`,
  `ScratchpadMemory`, `VpuUnit`, `DmaUnit`, `LutUnit`
- sources: matching `.cc` files

## Current Gaps / Caveats

- No MPU implementation exists yet despite the reserved command-space enum.
- There is no higher-level composite "tile/SEU cluster" C++ object wiring
  `2x VPU + 1x MPU + 1x DMA + 1x LUT`; test configs instantiate devices
  individually.
- Control and data traffic are intentionally not separated in the current gem5
  topology; they share the same `SystemXBar`.
- `SpecializedExecutionUnit` default behavior is synthetic and test-oriented.
  Any real accelerator semantics will need subclass-specific overrides.
- `DmaUnit` currently handles only `dataType == 0`.
- `LutUnit` is still a first-pass timing model: single resource, fixed latency,
  no bank-level contention, and no explicit queue-depth/backpressure policy.
- `VSoftmax` already has the correct functional boundary, but internal
  substages are not yet exported as separate timing counters.

## Recommended Regression Set

For the current nonlinear and shared-LUT baseline, the minimum useful
regression set is:

- unit-style:
  - `tests/gem5/npu/testcases/vpu/unary_exp`
  - `tests/gem5/npu/testcases/vpu/softmax_ramp`
- system-style:
  - `tests/gem5/npu/system_basic`
  - `tests/gem5/npu/testcases/system/vpu_dual_release_gate`
  - `tests/gem5/npu/system_pipeline`
  - `tests/gem5/npu/system_multiport`

What each group is intended to catch:

- `unary_exp`: unary nonlinear path correctness plus LUT timing behavior
- `vpu_softmax`: softmax correctness plus LUT accounting
- `system_basic`: command queue routing, sync order, one linear op plus one LUT
  op
- `system_vpu_dual`: two VPU instances sharing one LUT
- `system_pipeline`: DMA + linear VPU stage + softmax stage
- `system_multiport`: multi-port submission with shared queue and shared LUT

When adding new nonlinear behavior, update at least one unit-level case and one
system-level case from the list above.

## Recommended Reading Order

If another agent needs to understand this code quickly, read in this order:

1. `tests/gem5/npu/utils/cmd/common.hh` for the macro-command bit layout.
2. `src/npu/MegaCmdQueue.cc` for queueing, sync-wait, and routing.
3. `src/npu/SpecializedExecutionUnit.hh/.cc` for the common execution
   state machine.
4. `src/npu/VpuUnit.cc` for the current NPU execution semantics.
5. `src/npu/DmaUnit.cc` for the custom DMA data path.
6. `tests/gem5/npu/configs/npu_test_system.py` for actual system wiring in
   tests.
