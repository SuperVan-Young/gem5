# Copyright (c) 2026
# All rights reserved.

OUTDIR := ../bin
TARGETS := 4rv_sync_stress_riscv
OUTS := $(addprefix $(OUTDIR)/,$(TARGETS))
CPPFLAGS := -I../../utils -I../../utils/cmd

.PHONY: all clean

all: $(OUTS)

$(OUTDIR)/%: %.c
	mkdir -p $(OUTDIR)
	riscv64-linux-gnu-g++ $(CPPFLAGS) -O0 -static -o $@ $<

clean:
	rm -f $(OUTS)
