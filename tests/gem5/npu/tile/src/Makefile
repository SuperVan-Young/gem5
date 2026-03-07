# Copyright (c) 2026
# All rights reserved.

OUTDIR := ../bin
OUT := $(OUTDIR)/tile_megacmdqueue_seu_riscv
SRC := tile_megacmdqueue_seu_riscv.c

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	mkdir -p $(OUTDIR)
	riscv64-linux-gnu-gcc -O0 -static -o $@ $<

clean:
	rm -f $(OUT)
