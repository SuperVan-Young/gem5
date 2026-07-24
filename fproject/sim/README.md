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

The shell entry runs gem5 in `${USER}.gem5`. The PPA target frequency remains
2 GHz, while cycle simulation is normalized to 1 GHz. Performance uses summed
NPU profile busy cycles; elapsed cycles are retained as a diagnostic.

Task files explicitly configure `tiling.q=128` and `tiling.kv=128`, mapped to
FlashAttention `BR=BC=128`. QK, online softmax, and PV are all invoked once per
query/KV tile; the summary records block counts and validates profile macro
counts derived from the runtime shape. A partial final query tile is padded to
BR=128 for the softmax primitive's fixed layout unit; the summary reports
`padded_q_rows`, while QK/PV retain the logical row count.

The hardware YAML intentionally carries two different SRAM/SPM capacities:

- `memory.sram.capacity_bytes=262144` is the 256 KiB physical capacity used by
  PPA evaluation.
- `simulation.spm.size_bytes=8388608` is a sufficiently large functional
  workspace used by cycle simulation.

The cycle simulator does not use the 8 MiB workspace as PPA capacity.

Each run creates an independent directory under `results/` containing the
input snapshots, resolved simulation configuration, gem5 stdout/stderr, raw
profile log, parsed JSON, interactive HTML, manifest, and performance summary.
