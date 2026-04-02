# Copyright (c) 2026
# All rights reserved.

OUTDIR := ../bin
OUT := $(OUTDIR)/mpu_proto_riscv
SRC := mpu_proto_riscv.c
CPPFLAGS := -I../../utils -I../../utils/cmd

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	mkdir -p $(OUTDIR)
	riscv64-linux-gnu-g++ $(CPPFLAGS) -O0 -static -std=c++17 -o $@ $<

clean:
	rm -f $(OUT)
