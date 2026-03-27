# NPU System-Level Test Plan

This file records a staged implementation plan for building increasingly
realistic tests under `tests/gem5/npu/`, starting from a minimal "control unit
+ SEU" integration check and ending at a more complex system-level test that is
close to the intended real deployment shape.

The current codebase constraints matter:

- there is no MPU implementation in `src/npu` yet.
- there is no composite C++ SimObject representing "SEU contains 2 NPU + 1 MPU
  + 1 DMA".
- the actual model is a set of independent devices attached to one shared
  `SystemXBar`.
- control traffic and data traffic currently share that same bus.

Therefore the plan below is written around what the code can test today:

- `MegaCmdQueue` as the control unit / command distributor.
- `VpuUnit` instances as the currently implemented NPU-like execution units.
- `DmaUnit` as the implemented data mover.
- `ScratchpadMemory` plus DRAM-backed `SimpleMemory` as observable storage.

The plan is deliberately layered so that each phase can be implemented,
debugged, and landed independently.

## Guiding Principles

When adding any new test in this plan, follow these rules:

- Prefer one new scenario per directory unless the scenario set is tightly
  related and shares one workload/config naturally.
- Each phase should have a clear automated pass condition.
- The workload should print a stable pass marker that Python verifies.
- The Python config should also validate gem5-side counters and drain state.
- Start with single-CPU control flow first, then add multi-CPU contention.
- Do not assume internal parallel execution inside `MegaCmdQueue`; it dispatches
  commands sequentially from the queue head.
- Avoid introducing MPU references into executable behavior until MPU exists.

## Overall Target

The final target is a test that approximates a realistic accelerator control
flow:

1. CPU submits commands through `MegaCmdQueue`.
2. commands target multiple devices by `device_type` / `device_id`.
3. DMA moves data between DRAM and SPM.
4. two VPU/NPU instances consume and update shared SPM state.
5. sync indicator commands enforce inter-device ordering.
6. command queue and all devices drain cleanly.
7. final memory contents match a software-computed expectation.

Because MPU does not exist yet, the final target is a "reality-approximate"
test, not a full architectural reproduction.

## Phase Structure

- Phase 0: groundwork and reusable helpers
- Phase 1: first-version integrated control + SEU test
- Phase 2: strengthened control-flow test with richer sync structure
- Phase 3: second-version system test with 2x VPU + DMA
- Phase 4: multi-port / multi-CPU control stress on top of Phase 3
- Phase 5: hardening, cleanup, and future MPU insertion points

## Phase 0: Groundwork

### Objective

Prepare the test tree so later phases can reuse helpers instead of cloning
logic into each workload/config.

### Existing Assets To Reuse

- `tests/gem5/npu/configs/npu_test_system.py`
- `tests/gem5/npu/utils/cmd/common.hh`
- `tests/gem5/npu/utils/npu_sync.hh`
- `tests/gem5/npu/utils/npu_mmio.hh`
- existing examples under:
  - `seu_multiport/`
  - `vpu/`
  - `dma/`
  - `4rv/`

### Work Items

1. Audit whether a new helper header is needed for "system scenario" commands.
   Likely candidates:
   - one helper for issuing VPU commands with masks/repetition
   - one helper for issuing DMA commands from compact scenario parameters
2. If the new helpers would significantly reduce duplication, create them under
   `tests/gem5/npu/utils/cmd/`.
3. Keep helpers narrowly scoped:
   - no speculative MPU helper yet
   - no generic abstraction that hides command layout too much
4. Decide the directory naming for system-level tests. Recommended options:
   - `tests/gem5/npu/system/`
   - `tests/gem5/npu/cluster/`

Recommended choice:

- use `tests/gem5/npu/system/`

Reason:

- it reads naturally as "whole-system NPU integration tests"
- it does not imply a specific future hardware packaging
- it can hold both single-CPU and multi-CPU scenarios later

### Deliverables

- optional helper header(s), only if they clearly reduce duplication
- new directory skeleton for later phases

### Exit Criteria

- later phases can share command-building code cleanly
- no behavior change to existing tests

## Phase 1: First-Version Integrated Control + SEU Test

### Objective

Build the first new test that validates the minimal end-to-end path:

- CPU
- `MegaCmdQueue`
- sync command handling
- one execution device
- queue/drain completion

This phase is about control correctness first, not rich data correctness.

### Why This Phase Exists

Current tests prove parts of the stack separately:

- `tile/` proves queue + SEU can work
- `sit/` proves sync-indicator behavior
- `seu/` proves SEU counters

But there is not yet one focused test that combines these into a single,
explicit "control unit drives SEU through ordered command flow" scenario.

### Recommended Test Shape

Create a new directory, for example:

- `tests/gem5/npu/system_basic/`

Files to add:

- `tests/gem5/npu/system_basic/test_system_basic.py`
- `tests/gem5/npu/system_basic/configs/system_basic.py`
- `tests/gem5/npu/system_basic/src/system_basic_riscv.c`
- `tests/gem5/npu/system_basic/src/Makefile`
- optional `.gitignore`

### Device Topology

- 1 CPU
- 1 `MegaCmdQueue`
- 1 `VpuUnit` or 1 `SpecializedExecutionUnit`
- optional `ScratchpadMemory`

Recommended choice:

- use `VpuUnit`, not raw `SpecializedExecutionUnit`

Reason:

- command routing is more realistic
- `device_type/device_id` checks are enforced
- later phases can extend this scenario without replacing the device model

### Scenario Design

The workload should submit a short ordered sequence such as:

1. launch sync wait for indicator `A`
2. launch VPU exec command 0
3. set sync indicator `A`
4. launch VPU exec command 1
5. `npu_cmd_sync_done()`

Expected behavior:

- queue head blocks on sync wait until the indicator is set
- command 0 dispatches only after sync release
- command 1 dispatches after command 0
- `sync_done` response is deferred until queue drain completes

### What To Validate In The Workload

The workload should:

- print a stable pass marker only if it reaches normal completion
- optionally poll for a small SPM-visible effect if `VpuUnit` with SPM is used
- avoid relying purely on timing delays

Minimal workload checks are acceptable in Phase 1.

### What To Validate In The Python Config

Validate at least:

- `exit_cause == "exiting with last active thread context"`
- `cmdq.queueOccupancy() == 0`
- target execution unit `queueOccupancy() == 0`
- target execution unit `completedCmdCount()` equals expected command count
- target execution unit is not busy at end

If `VpuUnit` is used, also validate:

- `prologueCount()`
- `executeCount()`
- `epilogueCount()`

### Implementation Notes

- map command queue MMIO region
- map sync MMIO region
- if using SPM-backed validation, map SPM region too
- debug flags for first bring-up:
  - `MegaCmdQueue`
  - `VPU`
  - optionally `SpecializedExecutionUnit`

### Exit Criteria

- the test passes reliably with one clear PASS marker
- failures are diagnosable by queue occupancy and per-device counters
- the test demonstrates actual control ordering, not just command submission

## Phase 2: Richer Single-CPU Control-Flow Test

### Objective

Strengthen the first-version test so that it covers more realistic command
ordering patterns before introducing DMA.

### Why This Phase Exists

Going directly from Phase 1 to DMA + dual-VPU makes debugging harder. This
phase isolates control complexity from data-movement complexity.

### Scenario Upgrade

Extend the single-CPU sequence to include:

- multiple sync waits on different indicators
- multiple VPU commands with different masks and repetitions
- at least one command targeting `device_id = 1` if a second VPU is added early

Two good options:

1. keep one VPU and add several sync segments
2. add a second VPU but still no DMA

Recommended choice:

- add a second `VpuUnit` in this phase

Reason:

- it exercises command routing by `device_id`
- it is still easier to debug than involving DMA
- it directly builds toward the final target shape

### Recommended Topology

- 1 CPU
- 1 `MegaCmdQueue`
- 2 `VpuUnit` instances (`device_id = 0` and `1`)
- 1 `ScratchpadMemory`

### Scenario Pattern

For example:

1. sync wait `A`
2. VPU1 command
3. set sync `A`
4. VPU0 command
5. sync wait `B`
6. VPU1 command
7. set sync `B`
8. VPU0 command
9. `sync_done`

### Validation Focus

- routing to both VPU instances
- sync waits only unblock when expected
- counters for `vpu0` and `vpu1` match per-device expectations
- final SPM state matches software-computed expectation

### Deliverables

Either:

- evolve the Phase 1 test into a richer scenario

Or:

- add a second scenario in the same directory if the fixture structure is still
  manageable

Recommended choice:

- keep Phase 1 test stable and add a second scenario under a new directory such
  as `tests/gem5/npu/system_vpu_dual/`

### Exit Criteria

- deterministic final SPM contents
- correct per-VPU counters
- queue/drain clean at end

## Phase 3: Second-Version System Test With 2x VPU + DMA

### Objective

Build the first genuinely system-level scenario close to the intended hardware
usage, excluding MPU:

- control unit
- two NPU/VPU units
- one DMA
- shared SPM
- sync-ordered inter-device pipeline

This is the main "Version 2" milestone.

### Recommended Directory

- `tests/gem5/npu/system_pipeline/`

Files:

- `test_system_pipeline.py`
- `configs/system_pipeline.py`
- `src/system_pipeline_riscv.c`
- `src/Makefile`

### Device Topology

- 1 CPU
- 1 `MegaCmdQueue`
- 2 `VpuUnit` instances
- 1 `DmaUnit`
- 1 `ScratchpadMemory`
- DRAM-backed `SimpleMemory`

### Recommended Scenario

Use a simple but fully chained pipeline:

1. initialize a source tensor in DRAM
2. DMA command: DRAM -> SPM
3. sync wait on DMA completion indicator
4. VPU0 command: read/write selected SPM slots
5. sync wait on VPU0 completion indicator
6. VPU1 command: read/write selected SPM slots
7. optional DMA command: SPM -> DRAM
8. sync wait on final DMA completion indicator
9. `npu_cmd_sync_done()`
10. CPU verifies final SPM or DRAM contents

### Strong Recommendation

The first implementation of this phase should validate final SPM contents before
adding the final SPM -> DRAM copy-back.

Reason:

- one less stage to debug initially
- easier to inspect the exact intermediate data state

Suggested sub-steps:

- Phase 3A: DMA -> VPU0 -> VPU1, verify SPM
- Phase 3B: add final DMA copy-back, verify DRAM

### Data Strategy

Do not invent complicated tensor math here. Keep the data contract simple:

- let DMA copy known bytes into SPM
- let VPU0 modify a predictable subset
- let VPU1 modify another predictable subset or apply another known transform
- compute expected result in software inside the workload

Two possible styles:

1. tensor-style DMA plus slot-style VPU interaction
2. use DMA only to seed specific SPM slot-aligned data that VPU tests can read

Recommended choice:

- use DMA to populate a compact, deliberately chosen SPM region compatible with
  what the VPU commands will inspect

This keeps the scenario realistic but not overly broad.

### Python-Side Validation

Validate:

- exit cause and exit code
- `cmdq.queueOccupancy() == 0`
- `vpu0.queueOccupancy() == 0`
- `vpu1.queueOccupancy() == 0`
- `dma.queueOccupancy() == 0` if meaningful in the current API surface
- `vpu0.completedCmdCount() == expected`
- `vpu1.completedCmdCount() == expected`
- `cmdq` drain completed via `sync_done`

Also validate per-device counters if they remain stable:

- `prologueCount`
- `executeCount`
- `epilogueCount`
- `completedReadRespCount`
- `completedWriteRespCount`

### Workload-Side Validation

The workload should:

- compute the expected final state in software
- inspect the final SPM or DRAM contents
- print a stable PASS marker with scenario name

### Debug Bring-Up Order

When first implementing Phase 3, enable:

- `MegaCmdQueue`
- `DmaUnit`
- `VPU`
- `ScratchpadMemory`

### Exit Criteria

- one fully chained scenario passes reliably
- failures can distinguish DMA failure, sync failure, and VPU failure
- this test becomes the reference "near-real" system test in the tree

## Phase 4: Multi-Port / Multi-CPU Extension

### Objective

Take the Phase 3 pipeline and add realistic control contention:

- multiple CPU submitters
- multiple `MegaCmdQueue` input ports
- shared downstream device set

### Why This Phase Exists

The hardware intent includes a shared control fabric. Current tests have
separate pieces of this (`4rv/` for stress, system-style tests for pipeline),
but not the combined picture.

### Recommended Directory

- `tests/gem5/npu/system_multiport/`

### Topology

- 2 or 4 CPUs
- `MegaCmdQueue(num_input_port = num_cpus)`
- 2 `VpuUnit`
- 1 `DmaUnit`
- 1 `ScratchpadMemory`

### Scenario Options

Option A:

- one CPU acts as "DMA/control producer"
- one CPU acts as "VPU producer"

Option B:

- each CPU owns a phase of the pipeline and submits through its own port

Option C:

- all CPUs submit repeated mixed commands and use sync indicators for ownership

Recommended choice:

- start with Option A

Reason:

- easier to reason about correctness
- still exercises per-port mapping and queue admission
- incremental from Phase 3

### Validation Focus

- per-port command MMIO mapping works
- `sync_done` on the submitting port blocks until global drain
- queue drain remains correct under multi-port submission
- final memory state still matches expected result

### Exit Criteria

- deterministic correctness under multi-port submission
- no stuck queue head
- no residual device busy state

## Phase 5: Hardening And Future-Proofing

### Objective

Make the test suite durable and ready for later MPU insertion.

### Work Items

1. Consolidate helper code created in earlier phases.
2. Remove ad hoc duplicated command builders from workloads where practical.
3. Ensure each new test has:
   - one stable pass regex
   - clear scenario name
   - preserved artifacts through standard harness flow
4. Add README updates if the new tests introduce new conventions.
5. Mark clear extension points for a future MPU phase:
   - where a command sequence would gain MPU participation
   - which sync boundaries should remain unchanged

### Suggested Documentation Addition

When Phase 3 lands, update:

- `tests/gem5/npu/README.md`

to mention the new system-level scenario as the primary integration reference.

### Exit Criteria

- the new tests are maintainable, not one-off bring-up code
- future MPU support can be inserted by extending scenarios rather than
  redesigning them

## Recommended Implementation Order

If another agent picks this up, implement in this order:

1. Phase 0 skeleton and helper decision
2. Phase 1 single-CPU control + VPU test
3. Phase 2 dual-VPU single-CPU ordered-sync test
4. Phase 3A DMA -> VPU0 -> VPU1 with SPM validation
5. Phase 3B optional DMA copy-back to DRAM
6. Phase 4 multi-port extension
7. Phase 5 cleanup and documentation updates

## Concrete Acceptance Checklist By Version

### Version 1 Acceptance

Version 1 is considered complete when all of the following are true:

- there is one new explicit system/integration test beyond existing `tile/`
  and `sit/`
- it uses `MegaCmdQueue` plus at least one execution device
- it includes sync wait / sync release in the actual command stream
- Python validates drain state and device counters
- workload emits a stable PASS marker

### Version 2 Acceptance

Version 2 is considered complete when all of the following are true:

- one test includes `MegaCmdQueue + 2x VpuUnit + DmaUnit`
- command routing uses `device_type/device_id` across multiple devices
- at least one DMA completion gates a later VPU command via sync indicator
- final SPM or DRAM contents are checked against a software expectation
- all queues drain cleanly
- the scenario is documented as the closest current approximation to the real
  intended control/data flow without MPU

## Open Design Decisions

The following choices should be made explicitly before implementing Phase 1 or
Phase 3:

1. Directory naming:
   - recommended: `system_basic`, `system_vpu_dual`, `system_pipeline`
2. Whether to add new helper headers:
   - recommended: only if repeated command-packing logic becomes noisy
3. Whether Phase 1 should use SPM data validation:
   - recommended: optional, keep Phase 1 mostly control-focused
4. Whether Phase 3 should verify SPM first or DRAM first:
   - recommended: verify SPM first, then add DRAM copy-back

## Summary For A Future Agent

If you only read one section, read this:

- First build a simple single-CPU `MegaCmdQueue -> VPU` test with sync wait and
  drain validation.
- Then extend to dual VPU routing and richer sync structure.
- Then build the real milestone: `MegaCmdQueue + DMA + VPU0 + VPU1 + SPM`,
  ordered by sync indicators and validated by final memory contents.
- Only after that add multi-CPU / multi-port contention.

That sequence minimizes debug cost while steadily moving toward a realistic
system-level test.
