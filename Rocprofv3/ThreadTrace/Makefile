HIPCC ?= hipcc
TARGET ?= matmul
SRC ?= main.hip
ARCH ?= gfx942

# -g -gdwarf-4: source correlation in rocprofv3 ATT / stats CSV (see README).
CXXFLAGS ?= -O3 -std=c++17 -g -gdwarf-4 --offload-arch=$(ARCH) \
            -mllvm -amdgpu-early-inline-all=true

.PHONY: all clean run trace

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCC) $(CXXFLAGS) -o $@ $<

clean:
	rm -rf $(TARGET)  ui_output_agent_* stats_ui_*.csv

run: all
	./matmul

trace: all
	#rocprofv3 --att --att-activity 8 --kernel-iteration-range 6 --kernel-include-regex matmul_naive -- ./matmul
	#rocprofv3 --att --att-activity 8 --kernel-iteration-range 6 --kernel-include-regex matmul_mid -- ./matmul
	#rocprofv3 --att --att-activity 8 --kernel-iteration-range 6 --kernel-include-regex matmul_fast -- ./matmul
	rocprofv3 --att --att-activity 8 --kernel-iteration-range 6 -- ./matmul
