RISCV_CXX ?= riscv64-linux-gnu-g++
CXXFLAGS ?= -I../../../utils -I../../../utils/cmd -I../../../utils/golden \
	-O0 -static -std=c++17

BIN_DIR := bin
TARGET := $(BIN_DIR)/mpu_basic_gemm_tiled_riscv
SOURCES := workload.c ../../../utils/golden/mpu_gemm.cc

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) | $(BIN_DIR)
	$(RISCV_CXX) $(CXXFLAGS) -o $@ $(SOURCES)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)
