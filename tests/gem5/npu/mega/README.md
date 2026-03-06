# MegaCmdQueue test

This test verifies the basic MMIO behavior of `MegaCmdQueue` in a small SE-mode RISC-V setup.

## What it checks automatically

The test uses `tests/gem5/`'s standard harness and verifies stable output markers for:

- successful test completion,
- the expected process exit cause,
- the expected final queue occupancy.

A debug-log rejection event is also checked to confirm the full-queue path is exercised.

## Canonical way to run

From the repository root, use the gem5 test harness:

```bash
cd tests
./main.py run gem5/npu/mega
```

Per repository policy, this should normally be run inside the gem5 docker environment.

## Human inspection

Detailed run artifacts are preserved by the test harness under `tests/testing-results/` for manual inspection.

The test also enables the `MegaCmdQueue` debug flag so detailed behavior remains available when debugging failures.
