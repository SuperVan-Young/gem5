# Copyright (c) 2026
# All rights reserved.

OUTDIR := ../bin
OUT := $(OUTDIR)/dma_proto_riscv
SRC := dma_proto_riscv.c
CXX := riscv64-linux-gnu-g++
CXXFLAGS := -O0 -static -std=c++17 -x c++ -I../../utils

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	mkdir -p $(OUTDIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(OUT)
