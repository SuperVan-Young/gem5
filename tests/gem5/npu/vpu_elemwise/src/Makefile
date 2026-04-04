# Copyright (c) 2026
# All rights reserved.

OUTDIR := ../bin
OUT := $(OUTDIR)/vpu_elemwise_riscv
SRC := vpu_elemwise_riscv.c
CPPFLAGS := -I../../utils -I../../utils/cmd

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	mkdir -p $(OUTDIR)
	riscv64-linux-gnu-g++ $(CPPFLAGS) -O0 -static -o $@ $<

clean:
	rm -f $(OUT)
