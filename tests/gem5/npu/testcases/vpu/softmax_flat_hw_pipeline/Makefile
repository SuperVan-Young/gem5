# Copyright (c) 2026
# All rights reserved.

OUTDIR := bin
OUT := $(OUTDIR)/vpu_softmax_flat_hw_pipeline_riscv
SRC := workload.c ../../../utils/golden/vpu_softmax.cc
CXX := riscv64-linux-gnu-g++
CPPFLAGS := -I../../../utils -I../../../utils/cmd -I../../../utils/golden

HEADERS := $(wildcard ../../../utils/*.hh) \
	$(wildcard ../../../utils/cmd/*.hh) \
	$(wildcard ../../../utils/golden/*.hh)

WORKLOAD_OPT ?= -O2

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC) $(HEADERS)
	mkdir -p $(OUTDIR)
	$(CXX) $(CPPFLAGS) $(WORKLOAD_OPT) -static -o $@ $(SRC)

clean:
	rm -f $(OUT)
