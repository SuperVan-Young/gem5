# OnlineSoftmax

Run the default FlashAttention shape from the host with:

```sh
./fproject-atlas/sim/softmax/run.sh \
  --hardware fproject-atlas/configs/attention_chip.yaml \
  --rows 256 --cols 1024
```

The naive model follows ATLAS's edge-softmax convention exactly:

```text
vec_count = 9 * rows * cols
cycles = ceil(vec_count / vector_vec_num)
```

Here `rows` corresponds to ATLAS's
`batch_size * kv_head_num * kv_group_num`, while `cols` is its
`context_length`. The gem5 value is the sum of the 14 profiled VPU macro
durations. Raw profile, parsed JSON, interactive HTML, and the comparison are
written under `fproject-atlas/results/softmax/`.
