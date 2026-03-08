# Copyright (c) 2026
# All rights reserved.

OUTDIR := ../bin
TARGETS := sit_sync_indicator_riscv sit_launch_sync_launch_sync_riscv
OUTS := $(addprefix $(OUTDIR)/,$(TARGETS))

.PHONY: all clean

all: $(OUTS)

$(OUTDIR)/%: %.c
	mkdir -p $(OUTDIR)
	riscv64-linux-gnu-gcc -O0 -static -o $@ $<

clean:
	rm -f $(OUTS)
