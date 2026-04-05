# Copyright (c) 2026
# All rights reserved.

OUTDIR := bin
OUT := $(OUTDIR)/vpu_dual_release_gate_riscv
CXXSRC := workload.c ../../../utils/golden/vpu_unary.cc
CXX := riscv64-linux-gnu-g++
CPPFLAGS := -I../../../utils -I../../../utils/cmd -I../../../utils/golden
LDFLAGS := -lm

.PHONY: all clean

all: $(OUT)

$(OUT): $(CXXSRC)
	mkdir -p $(OUTDIR)
	$(CXX) $(CPPFLAGS) -O0 -static -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(OUT)
