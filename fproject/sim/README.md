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

Each run creates an independent directory under `results/` containing the
input snapshots, resolved simulation configuration, gem5 stdout/stderr, raw
profile log, parsed JSON, interactive HTML, manifest, and performance summary.
