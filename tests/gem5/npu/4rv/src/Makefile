# Copyright (c) 2026
# All rights reserved.

OUTDIR := ../bin
TARGETS := 4rv_sync_stress_riscv
OUTS := $(addprefix $(OUTDIR)/,$(TARGETS))

.PHONY: all clean

all: $(OUTS)

$(OUTDIR)/%: %.c
	mkdir -p $(OUTDIR)
	riscv64-linux-gnu-gcc -O0 -static -o $@ $<

clean:
	rm -f $(OUTS)
