OUTDIR := bin
OUT := $(OUTDIR)/atlas_matmul_riscv
SRC := workload.cc
CXX := riscv64-linux-gnu-g++
UTILS := ../../../tests/gem5/npu/utils
CPPFLAGS := -I$(UTILS) -I$(UTILS)/primitive -I$(UTILS)/cmd
HEADERS := $(wildcard $(UTILS)/*.hh) \
	$(wildcard $(UTILS)/primitive/*.hh) \
	$(wildcard $(UTILS)/cmd/*.hh)
WORKLOAD_OPT ?= -O2

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC) $(HEADERS)
	mkdir -p $(OUTDIR)
	$(CXX) $(CPPFLAGS) $(WORKLOAD_OPT) -static -std=c++17 -o $@ $(SRC)

clean:
	rm -f $(OUT)
