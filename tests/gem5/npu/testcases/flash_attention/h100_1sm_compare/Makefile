# Copyright (c) 2026
# All rights reserved.

OUTDIR := bin
OUT := $(OUTDIR)/flash_attention_h100_1sm_compare_riscv
SRC := workload.c ../../../../../../util/m5/src/abi/riscv/m5op.S
CXX := riscv64-linux-gnu-g++
CPPFLAGS := -I../../../utils -I../../../utils/golden \
	-I../../../utils/primitive -I../../../utils/cmd \
	-I../../../../../../include
LDFLAGS := -lm

HEADERS := $(wildcard ../../../utils/*.hh) \
	$(wildcard ../../../utils/golden/*.hh) \
	$(wildcard ../../../utils/primitive/*.hh) \
	$(wildcard ../../../utils/cmd/*.hh)

WORKLOAD_OPT ?= -O2

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC) $(HEADERS)
	mkdir -p $(OUTDIR)
	$(CXX) $(CPPFLAGS) $(WORKLOAD_OPT) -static -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(OUT)
