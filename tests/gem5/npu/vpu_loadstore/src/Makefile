OUTDIR := ../bin
OUT := $(OUTDIR)/vpu_loadstore_riscv
SRC := vpu_loadstore_riscv.c
CPPFLAGS := -I../../utils -I../../utils/cmd

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	mkdir -p $(OUTDIR)
	riscv64-linux-gnu-g++ $(CPPFLAGS) -O0 -static -o $@ $<

clean:
	rm -f $(OUT)
