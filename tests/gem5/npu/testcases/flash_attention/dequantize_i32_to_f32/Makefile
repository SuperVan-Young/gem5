# Copyright (c) 2026
# All rights reserved.

OUTDIR := bin
OUT := $(OUTDIR)/flash_attention_dequantize_i32_to_f32_riscv
SRC := workload.c
CXX := riscv64-linux-gnu-g++
CPPFLAGS := -I../../../utils -I../../../utils/primitive
LDFLAGS := -lm

HEADERS := $(wildcard ../../../utils/*.hh) \
	$(wildcard ../../../utils/primitive/*.hh)

WORKLOAD_OPT ?= -O2

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC) $(HEADERS)
	mkdir -p $(OUTDIR)
	$(CXX) $(CPPFLAGS) $(WORKLOAD_OPT) -static -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(OUT)
