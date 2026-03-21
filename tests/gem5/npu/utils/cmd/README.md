# Command Views

`common.h` is the base command helper for NPU macro instructions.
It defines the shared macro-command header, low-level bit access helpers, and launch helpers.

Add extra headers in this directory only when a command view has stabilized enough that dedicated `getX()` / `setX()` helpers reduce testcase complexity.
