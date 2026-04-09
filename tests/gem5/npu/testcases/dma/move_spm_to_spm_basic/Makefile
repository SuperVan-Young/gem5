# Copyright (c) 2026
# All rights reserved.

CASE_NAME := $(notdir $(CURDIR))
CASE_TAG := $(shell printf '%s' $(CASE_NAME) | tr '[:lower:]' '[:upper:]')

OUTDIR := bin
OUT := $(OUTDIR)/dma_$(CASE_NAME)_riscv
SRC := workload.c
CXX := riscv64-linux-gnu-g++
CPPFLAGS := -I../../../utils -I../../../utils/cmd \
	-DCASE_NAME_STR=\"$(CASE_NAME)\" \
	-DCASE_PASS_MARKER=\"DMA_$(CASE_TAG)_PASS\"

WORKLOAD_OPT ?= -O2

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	mkdir -p $(OUTDIR)
	$(CXX) $(CPPFLAGS) $(WORKLOAD_OPT) -static -o $@ $<

clean:
	rm -f $(OUT)
