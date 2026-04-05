OUTDIR := ../bin
OUT := $(OUTDIR)/vpu_reduce_riscv
SRC := vpu_reduce_riscv.c
CPPFLAGS := -I../../utils -I../../utils/cmd
LDFLAGS := -lm

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	mkdir -p $(OUTDIR)
	riscv64-linux-gnu-g++ $(CPPFLAGS) -O0 -static -o $@ $< $(LDFLAGS)

clean:
	rm -f $(OUT)
