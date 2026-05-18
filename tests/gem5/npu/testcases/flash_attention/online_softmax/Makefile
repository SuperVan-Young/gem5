# Copyright (c) 2026
# All rights reserved.

OUTDIR := bin
OUT := $(OUTDIR)/flash_attention_online_softmax_riscv
SRC := workload.c ../../../utils/golden/llm_f32.cc
CXX := riscv64-linux-gnu-g++
CPPFLAGS := -I../../../utils -I../../../utils/golden -I../../../utils/primitive
LDFLAGS := -lm

HEADERS := $(wildcard ../../../utils/*.hh) \
	$(wildcard ../../../utils/golden/*.hh) \
	$(wildcard ../../../utils/primitive/*.hh)

WORKLOAD_OPT ?= -O2

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC) $(HEADERS)
	mkdir -p $(OUTDIR)
	$(CXX) $(CPPFLAGS) $(WORKLOAD_OPT) -static -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(OUT)
