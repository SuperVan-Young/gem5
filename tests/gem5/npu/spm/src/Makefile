OUTDIR := ../bin
OUT := $(OUTDIR)/spm_test_riscv
SRC := spm_test.c
RISCV_PREFIX := /home/wangyukun/opt/riscv
CC := $(RISCV_PREFIX)/bin/riscv64-unknown-elf-gcc
export PATH := $(RISCV_PREFIX)/bin:$(PATH)

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	mkdir -p $(OUTDIR)
	$(CC) -O0 -static -o $@ $<

clean:
	rm -f $(OUT)
