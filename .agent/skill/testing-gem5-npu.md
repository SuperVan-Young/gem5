# testing-gem5-npu-v2 skill

## Purpose

Use this skill when adding, migrating, reviewing, or refactoring tests under
`tests/gem5/npu/`.

This skill is a specialization of `testing-gem5.md`. It defines the current V2
checked-in layout, testcase template, and migration-era rules for the NPU test
tree.

## Scope

Apply this skill when the task touches:

- `tests/gem5/npu/testcases/`
- `tests/gem5/npu/configs/`
- `tests/gem5/npu/configs/runner_common.py`
- `tests/gem5/npu/utils/`
- legacy shim files under `tests/gem5/npu/<old-module>/test_*.py`

Do not use this skill for generic gem5 tests outside `tests/gem5/npu/` unless
the user explicitly asks for NPU-specific structure.

## V2 mental model

An NPU testcase is split into three layers:

1. `test.py`
2. `config.py`
3. `workload.c`

The supporting public layers are:

- `tests/gem5/npu/configs/runner_common.py`
- `tests/gem5/npu/configs/npu_test_system.py`
- `tests/gem5/npu/configs/npu_test_common.py`
- `tests/gem5/npu/utils/`

Important current-state note:

- The checked-in tree uses `tests/gem5/npu/utils/` for shared workload helper
  headers.
- New testcase `Makefile`s currently include headers from `../../../utils` and
  `../../../utils/cmd`.
- `software_utils/` is still only a planned rename, not part of the current
  repository layout.

## Target directory structure

Top-level current structure for `tests/gem5/npu/`:

- `configs/`
- `utils/`
- `testcases/`

Within `testcases/`, organize:

1. first by module,
2. then by testcase.

Examples:

- `tests/gem5/npu/testcases/megacmdqueue/basic_mmio/`
- `tests/gem5/npu/testcases/megacmdqueue/sync_indicator/`
- `tests/gem5/npu/testcases/megacmdqueue/4rv_sync_stress/`
- `tests/gem5/npu/testcases/seu/basic/`

## Required testcase template

Each V2 testcase directory should stay flat and contain exactly these primary
files:

- `test.py`
- `config.py`
- `Makefile`
- `workload.c`
- `README.md`

Avoid adding a testcase-local `src/` directory in V2.

## File responsibilities

### `test.py`

`test.py` is the harness entry and the gem5 discovery entry.

It should:

- resolve the testcase-local config path,
- resolve the testcase-local binary path,
- construct the testcase-local build fixture,
- register `gem5_verify_config(...)`,
- expand scenarios when necessary.

It should not:

- build gem5 systems,
- inspect runtime stats,
- encode testcase-specific pass/fail semantics beyond verifiers.

Prefer `tests/gem5/npu/configs/runner_common.py` helpers:

- `resolve_config_path(...)`
- `resolve_binary_path(...)`
- `resolve_testcase_profile_path(...)`
- `make_binary_config_args(...)`
- `make_profile_gem5_args(...)`
- `make_testcase_build_fixture(...)`
- `register_npu_test(...)`
- `register_npu_scenarios(...)`

#### Primitive testcase profiling rule

All testcase directories under `tests/gem5/npu/testcases/primitive/` must
export NPU profiling artifacts by default.

Required pattern:

1. In `test.py`, set `gem5_args=make_profile_gem5_args(__file__)`.
2. In `test.py`, include `make_profile_artifact_verifier(__file__)` in
   `verifier_specs` so the raw log is always converted to JSON and HTML after
   the run.
3. The artifact set must land in testcase-local `profile/` as:
   - `<testcase>.npu_profile.log`
   - `<testcase>.npu_profile.json`
   - `<testcase>.npu_profile.html`
4. Generated profiling artifacts must stay gitignored.
5. Reuse `tests/gem5/npu/tools/parse_npu_profile.py` and
   `tests/gem5/npu/tools/render_npu_profile.py`; do not invent a second parse
   or render path for primitive tests.
6. These tests assume gem5 was built with the default NPU profiling support
   enabled (`NPU_PROFILE`); do not add a second primitive-specific build mode.
7. Prefer a meaningful 2D workload shape for primitive tests; default to at
   least `128 x 128` unless runtime becomes prohibitive, and size SPM / local
   buffer configuration to match the chosen tensor footprint.

Treat this as part of the primitive testcase template, not an optional add-on.

### `config.py`

`config.py` is the simulation configuration entry.

Recommended shape:

1. `build_m5_system(args)`
2. `collect_simulation_result(...)`
3. `verify_simulation_result(...)`

`config.py` should:

- build the simulated system,
- instantiate root,
- run the simulation,
- emit stable key/value summary lines,
- print one stable PASS marker when expectations match.

`config.py` should not:

- duplicate harness registration logic,
- hide testcase-specific expectations inside generic helpers.

Prefer:

- `NPUTestSystemBuilder` for stable system construction,
- `npu_test_common.py` for snapshot collection and summary emission.

Practical rule:

- Keep builder/process/root objects explicitly reachable at module scope before
  `m5.instantiate()`. Some gem5 object graphs are sensitive to early loss of
  visibility.

### `Makefile`

`Makefile` should:

- compile `workload.c`,
- place the binary under local `bin/`,
- clean only local build outputs.

Do not make a testcase-local `Makefile` responsible for global cleanup.

### `workload.c`

`workload.c` should:

- express testcase-specific software behavior,
- use reusable NPU MMIO/command helpers,
- stay deterministic.

Prefer legal, minimal command payloads for the current hardware model.
Do not keep stale random payload words if the current `SEU`/`VPU` model now
interprets those words as read/write masks or repetition fields.

### `README.md`

`README.md` must be in Chinese.

Use this fixed structure:

- `测试目的`
- `仿真系统`
- `仿真程序`
- `预期行为`

## Legacy shim rule

During migration, legacy test entry files under old module directories may stay
as compatibility shims.

Recommended pattern:

- keep the old `test_*.py`,
- replace its body with a small `importlib.util` shim that loads the new
  `testcases/.../test.py`.

Do not keep two independent harness implementations alive for the same test.

## Validation checklist

After changing an NPU testcase, run at least:

1. `python3 -m py_compile ...` for the changed Python files.
2. `util/style.py -m ...` in the docker environment.
3. The relevant `tests/main.py run -j32 --skip-build ... -vvv` command in the
   docker environment.

If the testcase is part of a legacy directory with known unrelated failures,
use the suite UID to run only the migrated testcase.

## Current validated V2 examples

Use these as templates first:

- `tests/gem5/npu/testcases/megacmdqueue/basic_mmio/`
- `tests/gem5/npu/testcases/megacmdqueue/sync_indicator/`
- `tests/gem5/npu/testcases/megacmdqueue/4rv_sync_stress/`
- `tests/gem5/npu/testcases/seu/basic/`

When in doubt, copy one of these structures and only then generalize.
