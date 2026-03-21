# NPU C Helper Layout

- `npu_mmio.hh`: contiguous `uint32_t` MMIO read/write helpers.
- `cmd/common.hh`: shared macro-command container, common header fields, field get/set helpers, and launch helpers.
- `npu_sync.hh`: sync-wait and sync-set scene helpers built on `cmd/common.hh`.

Common macro-command header fields currently live in the first 32 bits of `cmd/common.hh` word 0:
- `device_type` at bits `[31:28]`
- `device_id` at bits `[27:24]`
- `op_code` at bits `[23:16]`
- `sync_indicator` at bits `[15:8]`
- `set_indicator_sns` at bit `[7]`
- `set_indicator_snd` at bit `[6]`
- reserved bits at `[5:0]`

Prefer named field setters/getters for common header bits. Add new files under `utils/cmd/` only when a command view becomes stable and clearly reused.
