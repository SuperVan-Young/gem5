# testing-gem5 skill

## Purpose

Use this skill when asked to design, review, debug, or refactor tests under `tests/gem5/` in the gem5 repository.

This skill focuses on gem5 integration/system tests and does not prioritize `tests/pyunit/` unless the user explicitly asks for it.

## Core mental model

Most gem5 tests are declared through `gem5_verify_config(...)` and are best understood as:

- a gem5 run configuration,
- optional fixtures to prepare resources or binaries,
- one or more verifiers that encode the minimum correctness contract,
- preserved output artifacts for human inspection.

## What a typical `tests/gem5/` test contains

A standard test usually includes:

1. a test name,
2. a config script path,
3. `config_args`,
4. optional fixtures,
5. verifiers,
6. metadata such as `valid_isas`, `valid_hosts`, `valid_variants`, and `length`.

Primary references:

- `tests/gem5/suite.py`
- `tests/gem5/verifier.py`
- `tests/gem5/fixture.py`

## Output handling rule

`gem5_verify_config(...)` already manages a temp output directory and gem5 redirection. The output artifacts are preserved under `tests/testing-results/...` through `TempdirFixture`.

Practical rule:
- Prefer the standard harness output flow.
- Do not add a parallel manual output path unless the user explicitly wants one.
- If human inspection is needed, direct users to `tests/testing-results/...`.

## Verifier design rule

Start with the smallest stable verifier surface.

Prefer:
- one stable pass marker,
- or one stable regex proving completion,
- or a golden output when the output is intentionally stable.

Avoid overfitting to debug-log wording unless that wording is intentionally part of the test contract.

## Config script design rule

A `tests/gem5/` config script should usually:
- build the simulated system,
- run the workload,
- print a few stable summary lines.

Avoid putting too much test-verdict logic into the config when the verifier can express the check more clearly.

## Workload design rule

For local test workloads, prefer deterministic behavior and explicit completion.

Avoid:
- infinite loops used only to keep the simulation alive,
- implicit stop behavior when normal completion is possible.

If the test is specifically about timeout or tick-limited behavior, then explicit tick-exit is fine.

## Fixture rule for locally built test binaries

For local workload binaries, automatic fixture-based construction is preferable to requiring a manual `make` step before running the harness.

Recommended pattern:
- source under a local `src/` directory,
- binary output under a local `bin/` directory,
- build driven automatically from the test's fixtures.

## Important compatibility note about `MakeFixture` / `MakeTarget`

When debugging local auto-build fixtures, remember:

- the `Fixture` base class comes from `ext/testlib.fixture.Fixture`,
- it does not provide hidden dependency APIs like `require` or `required_by`.

If a Make-based fixture path fails with an error like:
- `AttributeError: 'MakeTarget' object has no attribute 'require'`

then the problem is in the local fixture implementation, not necessarily in the test itself.

A compatible strategy is:
- let `MakeTarget.setup(testitem)` directly invoke its associated `MakeFixture` to build the requested target,
- avoid relying on a fixture dependency graph that does not exist in the current loader/runtime.

## Duplicate verifier naming pitfall

Multiple `verifier.MatchRegex(...)` instances in one suite can produce duplicate generated subtest names, because they share the same verifier class name.

If you need multiple regex-based verifiers, consider:
- distinct verifier subclasses, or
- reducing checks to a smaller, more stable set.

## Default checklist when editing a gem5 test

Before changing or adding a test, check:

- What config script is under test?
- What exact runtime arguments are required?
- Does the test need fixtures for resources or locally built binaries?
- What is the smallest reliable automated correctness condition?
- What artifacts should remain available for human inspection?
- What `valid_isas`, `valid_hosts`, and `length` tags should be set?
- Can the standard `tests/testing-results/...` output flow replace any custom debug script output path?
