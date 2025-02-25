# Compiler
CXX = hipcc

# Default path for hipstdpar (can be overridden by user)
HIPSTDPAR_PATH ?= /opt/rocm/include/thrust/system/hip/hipstdpar

# Compiler flags
CXXFLAGS = -O3 --offload-arch=gfx90a,gfx942 --hipstdpar --hipstdpar-path=$(HIPSTDPAR_PATH)

# Targets
TARGET = final_timed
SRC = hip_stdpar_timer.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
