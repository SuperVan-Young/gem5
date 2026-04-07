# Copyright (c) 2026
# All rights reserved.

REPO_ROOT := $(abspath ../../../../../..)
OUTDIR := out
BINDIR := bin
RUN_DIR := $(OUTDIR)/run

SRC := workload.c ../../../utils/golden/vpu_softmax.cc
WORKLOAD_BIN := $(BINDIR)/profiling_tile_riscv
CONFIG := $(abspath config.py)
GEM5_BIN := $(REPO_ROOT)/build/RISCV/gem5.opt
PROFILE_LOG := $(RUN_DIR)/npu_profile.log
SIMOUT := $(OUTDIR)/simout.txt
SIMERR := $(OUTDIR)/simerr.txt
EVENTS_JSON := $(OUTDIR)/profile_events.json
HTML := $(OUTDIR)/profile_gantt.html
PARSER := $(REPO_ROOT)/tests/gem5/npu/tools/parse_npu_profile.py
RENDERER := $(REPO_ROOT)/tests/gem5/npu/tools/render_npu_profile.py

CXX := riscv64-linux-gnu-g++
PYTHON := python3
CPPFLAGS := -I../../../utils -I../../../utils/cmd -I../../../utils/golden
LDFLAGS := -lm
GEM5_DEBUG_FLAGS := --debug-flags=NPUProfile --debug-file=$(notdir $(PROFILE_LOG))

.PHONY: all gem5 workload run parse html clean

all: html

gem5:
	cd $(REPO_ROOT) && scons -j32 NPU_PROFILE=1 build/RISCV/gem5.opt

workload: $(WORKLOAD_BIN)

$(WORKLOAD_BIN): $(SRC)
	mkdir -p $(BINDIR)
	$(CXX) $(CPPFLAGS) -O0 -static -o $@ $(SRC) $(LDFLAGS)

run: gem5 workload
	rm -rf $(RUN_DIR)
	mkdir -p $(RUN_DIR) $(OUTDIR)
	cd $(REPO_ROOT) && \
		$(GEM5_BIN) -d $(abspath $(RUN_DIR)) $(GEM5_DEBUG_FLAGS) \
		$(CONFIG) --binary $(abspath $(WORKLOAD_BIN)) \
		> $(abspath $(SIMOUT)) 2> $(abspath $(SIMERR))
	grep -q "PROFILE_TILE_CONFIG_PASS" $(SIMOUT)

parse: run
	$(PYTHON) $(PARSER) --input $(PROFILE_LOG) --output $(EVENTS_JSON)

html: parse
	$(PYTHON) $(RENDERER) --input $(EVENTS_JSON) --output $(HTML)

clean:
	rm -rf $(OUTDIR) $(BINDIR)
