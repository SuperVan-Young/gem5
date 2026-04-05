# Copyright (c) 2026
# All rights reserved.

OUTDIR := ../bin
OUT := $(OUTDIR)/system_vpu_dual_riscv
SRC := system_vpu_dual_riscv.c ../../utils/golden/vpu_unary.cc
CPPFLAGS := -I../../utils -I../../utils/cmd -I../../utils/golden
LDFLAGS := -lm

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	mkdir -p $(OUTDIR)
	riscv64-linux-gnu-g++ $(CPPFLAGS) -O0 -static -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(OUT)
