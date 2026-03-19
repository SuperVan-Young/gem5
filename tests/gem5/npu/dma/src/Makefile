# Copyright (c) 2026
# All rights reserved.

OUTDIR := ../bin
OUT := $(OUTDIR)/dma_proto_riscv
SRC := dma_proto_riscv.c

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	mkdir -p $(OUTDIR)
	riscv64-linux-gnu-gcc -O0 -static -o $@ $<

clean:
	rm -f $(OUT)
