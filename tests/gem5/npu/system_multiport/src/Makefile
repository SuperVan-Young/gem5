# Copyright (c) 2026
# All rights reserved.

OUTDIR := ../bin
OUT := $(OUTDIR)/system_multiport_riscv
SRC := system_multiport_riscv.c
RISCV_PREFIX := /home/wangyukun/opt/riscv
CC := $(RISCV_PREFIX)/bin/riscv64-unknown-elf-gcc
export PATH := $(RISCV_PREFIX)/bin:$(PATH)
CPPFLAGS := -I../../utils -I../../utils/cmd

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	mkdir -p $(OUTDIR)
	$(CC) -x c++ $(CPPFLAGS) -O0 -static -o $@ $<

clean:
	rm -f $(OUT)
