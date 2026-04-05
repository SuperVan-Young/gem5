# Copyright (c) 2026
# All rights reserved.

OUTDIR := bin
OUT := $(OUTDIR)/vpu_legacy_dual_device_basic_riscv
SRC := workload.c
CXX := riscv64-linux-gnu-g++
CPPFLAGS := -I../../../utils -I../../../utils/cmd

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	mkdir -p $(OUTDIR)
	$(CXX) $(CPPFLAGS) -O0 -static -o $@ $<

clean:
	rm -f $(OUT)
