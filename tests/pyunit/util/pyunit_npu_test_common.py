#!/usr/bin/env python3
#
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

import contextlib
import io
import sys
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root / "tests" / "gem5" / "npu" / "configs"))

from npu_test_common import (  # noqa: E402
    collect_component_snapshot,
    emit_summary,
    map_multi_cpu_regions,
    map_single_cpu_regions,
    verify_snapshot,
)


class _DummyComponent:
    def foo(self):
        return 3

    def bar(self):
        return 7


class _DummyBuilder:
    def __init__(self):
        self.calls = []
        self.processes = ["p0", "p1"]

    def get_process(self, cpu_id):
        return self.processes[cpu_id]

    def get_processes(self):
        return list(self.processes)

    def map_cmdq(self, process, cpu_id):
        self.calls.append(("cmdq", cpu_id, process))

    def map_seu(self, process, cpu_id):
        self.calls.append(("seu", cpu_id, process))

    def map_spm(self, process, cpu_id):
        self.calls.append(("spm", cpu_id, process))

    def map_sync(self, process, cpu_id):
        self.calls.append(("sync", cpu_id, process))

    def map_dram(self, process, cpu_id):
        self.calls.append(("dram", cpu_id, process))

    def map_cmdq_port(self, cpu_id, process):
        self.calls.append(("cmdq_port", cpu_id, process))


class NpuTestCommonTestCase(unittest.TestCase):
    def test_collect_component_snapshot(self):
        snapshot = collect_component_snapshot(
            _DummyComponent(),
            {
                "foo": "foo",
                "bar": "bar",
            },
        )

        self.assertEqual(snapshot, {"foo": 3, "bar": 7})

    def test_emit_summary(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            emit_summary("SEU", {"completed_cmds": 5, "issue_busy": False})

        self.assertEqual(
            buffer.getvalue().splitlines(),
            ["SEU_COMPLETED_CMDS=5", "SEU_ISSUE_BUSY=False"],
        )

    def test_map_single_cpu_regions(self):
        builder = _DummyBuilder()

        map_single_cpu_regions(
            builder,
            cpu_id=0,
            map_cmdq=True,
            map_seu=True,
            map_spm=True,
            map_sync=True,
            map_dram=True,
        )

        self.assertEqual(
            builder.calls,
            [
                ("cmdq", 0, "p0"),
                ("seu", 0, "p0"),
                ("spm", 0, "p0"),
                ("sync", 0, "p0"),
                ("dram", 0, "p0"),
            ],
        )

    def test_map_multi_cpu_regions(self):
        builder = _DummyBuilder()

        map_multi_cpu_regions(
            builder,
            processes=builder.get_processes(),
            map_cmdq_ports=True,
            map_spm=True,
            map_sync=True,
            map_dram=True,
        )

        self.assertEqual(
            builder.calls,
            [
                ("cmdq_port", 0, "p0"),
                ("spm", 0, "p0"),
                ("sync", 0, "p0"),
                ("dram", 0, "p0"),
                ("cmdq_port", 1, "p1"),
                ("spm", 1, "p1"),
                ("sync", 1, "p1"),
                ("dram", 1, "p1"),
            ],
        )

    def test_verify_snapshot(self):
        self.assertTrue(
            verify_snapshot(
                {"completed_cmds": 5, "issue_busy": False},
                {"completed_cmds": 5, "issue_busy": False},
            )
        )
        self.assertFalse(
            verify_snapshot(
                {"completed_cmds": 5, "issue_busy": False},
                {"completed_cmds": 4, "issue_busy": False},
            )
        )


if __name__ == "__main__":
    unittest.main()
