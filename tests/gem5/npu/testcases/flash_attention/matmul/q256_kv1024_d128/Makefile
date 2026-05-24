# Copyright (c) 2026
# All rights reserved.

OUTDIR := bin
OUT := $(OUTDIR)/flash_attention_matmul_q256_kv1024_d128_riscv
SRC := workload.c
CXX := riscv64-linux-gnu-g++
CPPFLAGS := -I../../../../utils -I../../../../utils/golden \
	-I../../../../utils/primitive -I../../../../utils/cmd
LDFLAGS := -lm

HEADERS := $(wildcard ../../../../utils/*.hh) \
	$(wildcard ../../../../utils/golden/*.hh) \
	$(wildcard ../../../../utils/primitive/*.hh) \
	$(wildcard ../../../../utils/cmd/*.hh)

WORKLOAD_OPT ?= -O2

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC) $(HEADERS)
	mkdir -p $(OUTDIR)
	$(CXX) $(CPPFLAGS) $(WORKLOAD_OPT) -static -std=c++17 -o $@ $(SRC) \
		$(LDFLAGS)

clean:
	rm -f $(OUT)
