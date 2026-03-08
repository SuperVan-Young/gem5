# Copyright (c) 2026
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from m5.objects.AbstractMemory import AbstractMemory
from m5.params import *


class ScratchpadMemory(AbstractMemory):
    """Scratchpad Memory (SPM) for accelerator data caching.

    This is a timing-mode memory device that simulates SRAM behavior
    with configurable latency and bandwidth. It is designed to be
    attached to an accelerator bus and provides fast local storage
    for accelerator units within a Tile.
    """

    type = "ScratchpadMemory"
    cxx_header = "npu/mega/ScratchpadMemory.hh"
    cxx_class = "gem5::npu::ScratchpadMemory"

    port = ResponsePort("This port sends responses and receives requests")

    # SRAM access latency (read and write have same latency for SRAM)
    latency = Param.Latency("10ns", "SRAM access latency")

    # Bandwidth limit for SRAM (typical SRAM bandwidth)
    # Default set to 100GiB/s for high-performance SRAM
    bandwidth = Param.MemoryBandwidth(
        "100GiB/s", "Combined read and write bandwidth"
    )
