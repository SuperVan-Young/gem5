# NPU + DMA Merge Debug Plan

## Scope

This note checks the current merge boundary between:

- the DMA implementation under `src/npu/mega/DmaUnit.*`
- the NPU-side system/test code that was written against the `src/npu/README.md`
  execution model

I did not build or run gem5 here. The issues below are from static inspection of
`src/npu`, `tests/gem5/npu`, and the current helper/workload code.

## Confirmed Merge Problems

### 1. Python builder API mismatch blocks DMA system configs before simulation

Location:

- [tests/gem5/npu/configs/npu_test_system.py](/home/wangyukun/workspace/gem5/tests/gem5/npu/configs/npu_test_system.py#L337)
- [tests/gem5/npu/configs/npu_test_system.py](/home/wangyukun/workspace/gem5/tests/gem5/npu/configs/npu_test_system.py#L414)

Observed mismatch:

- `add_dma()` accepts `bank_size`
- `add_mega_seu()` calls `add_dma(..., buffer_size=dma_buffer_size, ...)`

Impact:

- any config using `add_mega_seu(include_dma=True)` will fail in Python with an
  unexpected-keyword error before `m5.instantiate()`
- this directly affects at least:
  - `tests/gem5/npu/system_pipeline/configs/system_pipeline.py`
  - `tests/gem5/npu/system_multiport/configs/system_multiport.py`

Root cause:

- the builder still uses an older DMA parameter name (`buffer_size`) while the
  merged `DmaUnit` SimObject exposes `bank_size`

### 2. System DMA command helpers still encode an old DMA protocol

Reference for the current DMA command layout:

- [src/npu/mega/DmaUnit.cc](/home/wangyukun/workspace/gem5/src/npu/mega/DmaUnit.cc#L94)
- [tests/gem5/npu/dma/src/dma_proto_riscv.c](/home/wangyukun/workspace/gem5/tests/gem5/npu/dma/src/dma_proto_riscv.c#L235)

Stale helpers:

- [tests/gem5/npu/system_pipeline/src/system_pipeline_riscv.c](/home/wangyukun/workspace/gem5/tests/gem5/npu/system_pipeline/src/system_pipeline_riscv.c#L183)
- [tests/gem5/npu/system_multiport/src/system_multiport_riscv.c](/home/wangyukun/workspace/gem5/tests/gem5/npu/system_multiport/src/system_multiport_riscv.c#L200)

Problems inside those helpers:

- they write `src_base` to `words[14]` and `dst_base` to `words[13]`, but the
  merged DMA parser reads them from words `1` and `2`
- they do not populate the merged DMA shape/stride/block fields in the expected
  slots
- they do not build `mode_cfg`, so source/destination memory spaces and cut
  dimensions are not encoded correctly

Impact:

- even after fixing the Python builder, the system tests will still drive DMA
  with malformed commands
- the first DMA command can be misparsed, panic, or silently touch the wrong
  fields depending on the values

Root cause:

- system-level test workloads were not updated to the new 64-byte DMA command
  schema already used by `dma_proto_riscv.c`

### 3. The system helpers still use a pre-merge DMA mode meaning

Location:

- [tests/gem5/npu/system_pipeline/src/system_pipeline_riscv.c](/home/wangyukun/workspace/gem5/tests/gem5/npu/system_pipeline/src/system_pipeline_riscv.c#L14)
- [tests/gem5/npu/system_multiport/src/system_multiport_riscv.c](/home/wangyukun/workspace/gem5/tests/gem5/npu/system_multiport/src/system_multiport_riscv.c#L15)
- [src/npu/mega/DmaUnit.hh](/home/wangyukun/workspace/gem5/src/npu/mega/DmaUnit.hh)

Observed mismatch:

- system tests treat mode `0x0` as DRAM->SPM and mode `0x1` as SPM->DRAM
- merged `DmaUnit` treats mode bits as:
  - `0 = MoveLayout`
  - `1 = Transpose`
  - `2 = Fill`

Impact:

- the copy-back DMA in `system_pipeline` and `system_multiport` is currently
  encoded as mode `1`, which the merged DMA interprets as `Transpose`, not
  `SPM -> DRAM move_layout`
- direction now comes from `mode_cfg` memory-space bits, not the mode value

Root cause:

- direction-based DMA opcode encoding was not migrated to the merged
  move-layout encoding

### 4. DMA completion sync is not requested by the system helpers

Reference for completion-sync behavior:

- [src/npu/mega/SpecializedExecutionUnit.cc](/home/wangyukun/workspace/gem5/src/npu/mega/SpecializedExecutionUnit.cc#L867)
- [tests/gem5/npu/dma/src/dma_proto_riscv.c](/home/wangyukun/workspace/gem5/tests/gem5/npu/dma/src/dma_proto_riscv.c#L245)

Observed mismatch:

- the working DMA test helper sets `setSetIndicatorSns(...)`
- `system_pipeline_riscv.c` and `system_multiport_riscv.c` only set
  `syncIndicator`, then immediately issue `npu_launch_sync_wait(...)`

Impact:

- even with a correctly formatted DMA command, the queue wait command has no
  reason to unblock because DMA was not asked to raise the sync indicator on
  completion

Root cause:

- the system helpers kept the old submission pattern and missed the completion
  handshake required by the merged SEU/DMA flow

## Recommended Fix Order

### Step 1. Fix the builder-level API mismatch first

Change `tests/gem5/npu/configs/npu_test_system.py` so `add_mega_seu()` passes
the current DMA parameter name used by `add_dma()` and `DmaUnit.py`.

Goal:

- make `system_pipeline.py` and `system_multiport.py` instantiate again

Why first:

- until this is fixed, no DMA-inclusive system config can reach runtime

### Step 2. Replace the stale system DMA packing logic with one shared helper

Do not patch the two system workloads independently with another ad-hoc
`push_dma()` format.

Preferred direction:

- add a small DMA command helper next to the existing NPU helpers under
  `tests/gem5/npu/utils/`
- make `system_pipeline_riscv.c` and `system_multiport_riscv.c` build DMA
  commands through that helper
- use `dma_proto_riscv.c` as the ground truth for word placement and sync-bit
  handling

Minimum helper behavior:

- always emit mode `MoveLayout` for copy commands
- encode direction through `mode_cfg` source/destination memory-space bits
- write source/destination base addresses into words `1` and `2`
- write tensor shape/stride/block fields into words `3..14`
- optionally set `set_indicator_sns` when the caller needs a completion sync

Why shared:

- the same stale pattern exists in two tests already
- keeping a single helper reduces the chance of future DMA protocol drift

### Step 3. Update the two system workloads to the merged DMA semantics

Affected files:

- [tests/gem5/npu/system_pipeline/src/system_pipeline_riscv.c](/home/wangyukun/workspace/gem5/tests/gem5/npu/system_pipeline/src/system_pipeline_riscv.c)
- [tests/gem5/npu/system_multiport/src/system_multiport_riscv.c](/home/wangyukun/workspace/gem5/tests/gem5/npu/system_multiport/src/system_multiport_riscv.c)

Concrete changes needed:

- remove the old `DmaXferMode` direction encoding
- build both DMA copies as `MoveLayout`
- set source/destination memory spaces from the actual base addresses
- set `set_indicator_sns = 1` for the DMA commands that are followed by
  `npu_launch_sync_wait(...)`

Expected result:

- `system_pipeline` can again model `DMA -> VPU0 -> VPU1 -> optional DMA`
- `system_multiport` can again model multi-port enqueue with DMA + shared LUT

### Step 4. Keep DMA-only regression as the protocol oracle

Use `tests/gem5/npu/dma` as the reference path because it already matches the
merged `DmaUnit` layout much better than the two system tests.

Practical rule:

- if there is a conflict between `dma_proto_riscv.c` and the system test DMA
  packers, trust `dma_proto_riscv.c`

## Validation Plan For You

I cannot run the required gem5 build here. After the code fix, please validate
in this order.

### Rebuild the test workloads

For the system workloads that use `riscv64-unknown-elf-gcc`:

```sh
make -C tests/gem5/npu/system_pipeline/src
make -C tests/gem5/npu/system_multiport/src
```

If you also touch the DMA protocol reference workload:

```sh
make -C tests/gem5/npu/dma/src
```

### Rebuild gem5 in docker

Per repo rules:

```sh
docker exec -i "${USER}.gem5" bash -lc "cd /gem5 && scons -j32 build/RISCV/gem5.opt"
```

### Run the focused regressions in docker

Start with DMA protocol coverage:

```sh
docker exec -i "${USER}.gem5" bash -lc "cd /gem5/tests && ./main.py run -j32 --skip-build gem5/npu/dma -vvv"
```

Then run the two affected system regressions:

```sh
docker exec -i "${USER}.gem5" bash -lc "cd /gem5/tests && ./main.py run -j32 --skip-build gem5/npu/system_pipeline -vvv"
docker exec -i "${USER}.gem5" bash -lc "cd /gem5/tests && ./main.py run -j32 --skip-build gem5/npu/system_multiport -vvv"
```

If those pass, run the README-recommended system baseline:

```sh
docker exec -i "${USER}.gem5" bash -lc "cd /gem5/tests && ./main.py run -j32 --skip-build gem5/npu/system_basic -vvv"
docker exec -i "${USER}.gem5" bash -lc "cd /gem5/tests && ./main.py run -j32 --skip-build gem5/npu/system_vpu_dual -vvv"
```

## Expected Debugging Payoff

If the merge problems above are fixed in this order:

1. config construction failures should disappear first
2. DMA commands should become decodable again
3. sync-wait sequencing in the system tests should stop hanging or panicking
4. the remaining failures, if any, will be much more likely to be real DMA/VPU
   runtime issues rather than protocol mismatch noise
