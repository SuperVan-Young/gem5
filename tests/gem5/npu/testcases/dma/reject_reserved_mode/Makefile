# Copyright (c) 2026
# All rights reserved.

OUTDIR := bin
OUT := $(OUTDIR)/dma_reject_reserved_mode_riscv
SRC := workload.c
CXX := riscv64-linux-gnu-g++
CXXFLAGS := $(WORKLOAD_OPT) -static -std=c++17 -x c++ -I../../../utils -I../../../utils/cmd

WORKLOAD_OPT ?= -O2

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	mkdir -p $(OUTDIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(OUT)
