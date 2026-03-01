# Copyright (c) 2026
# All rights reserved.

import sys

import m5
from m5.objects import *


def expect(name, cond):
    if cond:
        print(f"PASS: {name}")
        return

    print(f"FAIL: {name}")
    sys.exit(1)


depth = 2
cmd_width = 128
chunk_bits = 32
words_per_cmd = cmd_width // chunk_bits

system = System(
    mem_mode="timing",
    membus=SystemXBar(),
    physmem=SimpleMemory(range=AddrRange("64MiB")),
    clk_domain=SrcClockDomain(clock="1GHz", voltage_domain=VoltageDomain()),
)
system.system_port = system.membus.cpu_side_ports
system.physmem.port = system.membus.mem_side_ports

system.cmdq = MegaCmdQueue(
    num_input_port=1,
    mega_cmd_width=cmd_width,
    cmd_queue_depth=depth,
)

root = Root(full_system=False, system=system)
m5.instantiate()

cmdq = root.system.cmdq

for cmd_idx in range(depth):
    for word_idx in range(words_per_cmd):
        value = (cmd_idx << 8) | word_idx
        expect(f"write cmd{cmd_idx} word{word_idx}", cmdq.testWriteWord(0, value))
    expect(f"doorbell cmd{cmd_idx}", cmdq.testRingDoorbell())

expect("queue full", cmdq.queueOccupancy() == depth)

for word_idx in range(words_per_cmd):
    expect(f"stage overflow-check word{word_idx}", cmdq.testWriteWord(0, 0xA500 + word_idx))

expect("reject when full", not cmdq.testRingDoorbell())
expect("pop succeeds", cmdq.popCmd())
expect("occupancy decremented", cmdq.queueOccupancy() == depth - 1)
expect("accept after pop", cmdq.testRingDoorbell())
expect("occupancy restored", cmdq.queueOccupancy() == depth)

print("MEGACMDQUEUE_TEST_PASS")
sys.exit(0)
