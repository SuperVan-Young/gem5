# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

gem5 is a modular computer architecture simulator.

Key parts of this repo:
- `src/`: C++ simulator core (built with SCons)
- `src/python/`: Python packages used by gem5 (incl. gem5 “standard library”)
- `configs/`: example simulation configuration scripts
- `tests/`: test harness and test suites (entry: `tests/main.py`)
- `util/`: developer utilities (style checker, hooks)

## Required workflow in this repo

### Use the docker environment only for build/test/format

```sh
docker exec -i "${USER}.gem5" bash -lc "cd /gem5 && <command>"
```

Use the docker environment if and only if you need to build, test, or run formatting/style checks.

For other tasks, such as code search, reading files, reviewing code, or general repository inspection, do not use the docker environment.

### Build (defaults)

- Always build with **`-j32`**.
- Default ISA target is **RISCV**.

Common build:
```sh
scons -j32 build/RISCV/gem5.opt
```

If you need SCons options/help:
```sh
scons -h
```

### Testing (required after changes)

After *each* code change, you must:
1) Add/extend **unit tests** covering the changed behavior.
2) **Build and run** the relevant unit tests to validate correctness.

For a single test, run:
```sh
cd tests
./main.py run -j32 --skip-build <test-folder-name> -vvv
```

If needed, use the debug output to find the generated run directory, then inspect `simout.txt` and `simerr.txt` there to help identify the root cause.

### Formatting / style checks (required)

Inside the docker environment, do **not** install extra dependencies.

After writing code, run formatting/style checks (typical choices in this repo):
```sh
pre-commit run --all-files
```

And/or run gem5’s style checker (whole-file or modified regions):
```sh
util/style.py
util/style.py -m
```

## Architecture & code structure (big picture)

### Build system layering

- `SConstruct` is the top-level entrypoint.
  - Defines build targets like `build/<ISA>/gem5.{debug,opt,fast}`.
  - Supports Kconfig-based configuration for build directories (e.g., `scons menuconfig <builddir>`).
  - Configures toolchain features (C++17, sanitizers, embedded Python, etc.).
- `src/SConscript` defines how sources are collected and built, including Python embedding/build rules.

Practical implication: when adding new C++ files or new SimObjects, you typically need to update the relevant `SConscript`/`SConsopts` files in that subtree so they’re included in the build.

### Python ↔ C++ boundary (SimObjects)

A central concept is the **SimObject**:
- SimObjects are configured in Python and instantiated/bound into the C++ simulator.
- The build generates parameter structures and related C++ artifacts from Python SimObject definitions (see `src/SConscript` for `SimObject`/`PySource` build rules).

Practical implication: adding/changing parameters commonly touches both Python and C++ plus the build glue that generates params.

## Testing guidance

For `tests/gem5/` structure, pass/fail conventions, build/run workflow, and the recommended pattern of automated verification plus preserved artifacts for human inspection, see:

- `.agent/skill/testing-gem5.md`

Prefer these notes when asked about writing, reviewing, or debugging gem5 tests. Unless explicitly requested by the user, do not retain or prioritize `tests/pyunit/` guidance.

## Commit conventions (repo-enforced)

- Commit headers must start with one or more **tags** (from `MAINTAINERS.yaml`) followed by a colon, e.g. `mem-ruby: ...` or `mem,mem-cache: ...`.
- Header line (tags + title) must be **≤ 65 characters**.
- If there’s a body, the header must be followed by an empty line.

These rules are enforced by the commit-msg hook (`util/git-commit-msg.py`) when pre-commit is installed.

