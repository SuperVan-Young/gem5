# Copyright (c) 2026
# All rights reserved.

OUTDIR := bin
OUT := $(OUTDIR)/flash_attention_layout_planner_riscv
SRC := workload.c
CXX := riscv64-linux-gnu-g++
CPPFLAGS := -I../../../utils

HEADERS := $(wildcard ../../../utils/*.hh)

WORKLOAD_OPT ?= -O2

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC) $(HEADERS)
	mkdir -p $(OUTDIR)
	$(CXX) $(CPPFLAGS) $(WORKLOAD_OPT) -static -o $@ $(SRC)

clean:
	rm -f $(OUT)
