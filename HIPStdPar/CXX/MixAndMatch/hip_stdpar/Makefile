# Compiler
CXX = hipcc

# Default path for hipstdpar (can be overridden by user)
STDPAR_PATH ?= /opt/rocm/include/thrust/system/hip/hipstdpar

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

# Compiler flags
CXXFLAGS = -O3 --offload-arch=${ROCM_GPU} --hipstdpar --hipstdpar-path=$(STDPAR_PATH)

# Targets
TARGET = final_timed
SRC = hip_stdpar_timer.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
