# Copyright (c) 2026
# All rights reserved.

OUTDIR := bin
CASE := fill_zero_dram
OUT := $(OUTDIR)/dma_$(CASE)_riscv
SRC := workload.c
CXX := riscv64-linux-gnu-g++
CPPFLAGS := -I../../../utils -I../../../utils/cmd

WORKLOAD_OPT ?= -O2

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	mkdir -p $(OUTDIR)
	$(CXX) $(CPPFLAGS) $(WORKLOAD_OPT) -static -std=c++17 -x c++ -o $@ $<

clean:
	rm -f $(OUT)
