OUTDIR := ../bin
OUT := $(OUTDIR)/vpu_loadstore_riscv
SRC := vpu_loadstore_riscv.c
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
