# Copyright (c) 2026
# All rights reserved.

OUTDIR := bin
OUT := $(OUTDIR)/vpu_legacy_dual_device_basic_riscv
SRC := workload.c
CXX := riscv64-linux-gnu-g++
CPPFLAGS := -I../../../utils -I../../../utils/cmd

HEADERS := $(wildcard ../../../utils/*.hh) \
	$(wildcard ../../../utils/cmd/*.hh) \
	$(wildcard ../../../utils/golden/*.hh)

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC) $(HEADERS)
	mkdir -p $(OUTDIR)
	$(CXX) $(CPPFLAGS) -O0 -static -o $@ $(SRC)

clean:
	rm -f $(OUT)
