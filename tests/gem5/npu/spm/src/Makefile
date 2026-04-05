OUTDIR := ../bin
OUT := $(OUTDIR)/spm_test_riscv
SRC := spm_test.c

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	mkdir -p $(OUTDIR)
	riscv64-linux-gnu-g++ -O0 -static -o $@ $<

clean:
	rm -f $(OUT)
