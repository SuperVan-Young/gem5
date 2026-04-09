# Copyright (c) 2026
# All rights reserved.

OUTDIR := bin
OUT := $(OUTDIR)/spm_test_riscv
SRC := workload.c
CC := /usr/bin/riscv64-linux-gnu-gcc

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	mkdir -p $(OUTDIR)
	$(CC) -O0 -static -o $@ $<

clean:
	rm -f $(OUT)
