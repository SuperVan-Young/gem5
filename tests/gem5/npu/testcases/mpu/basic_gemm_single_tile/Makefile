# Copyright (c) 2026
# All rights reserved.

OUTDIR := bin
OUT := $(OUTDIR)/mpu_basic_gemm_single_tile_riscv
SRC := workload.c ../../../utils/golden/mpu_gemm.cc
CXX := riscv64-linux-gnu-g++
CPPFLAGS := -I../../../utils -I../../../utils/cmd -I../../../utils/golden

HEADERS := $(wildcard ../../../utils/*.hh) \
	$(wildcard ../../../utils/cmd/*.hh) \
	$(wildcard ../../../utils/golden/*.hh)

WORKLOAD_OPT ?= -O0

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC) $(HEADERS)
	mkdir -p $(OUTDIR)
	$(CXX) $(CPPFLAGS) $(WORKLOAD_OPT) -static -std=c++17 -o $@ $(SRC)

clean:
	rm -f $(OUT)
