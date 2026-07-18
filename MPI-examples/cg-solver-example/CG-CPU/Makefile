# =============================================================================
# Makefile for the CPU Conjugate Gradient solver
#
# Usage:
#   module load rocm-*/ucx-*/ompi/*   # any MPI with mpicxx
#   make
#   mpirun -n 4 ./cg_cpu src/Dubcova2.pm
# =============================================================================

CXX      = mpicxx
CXXFLAGS = -O3 -std=c++17
INCLUDES = -Isrc

TARGET = cg_cpu
SRC    = src/cg.cpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

run: $(TARGET)
	mpirun -n 1 ./$(TARGET) src/Dubcova2.pm

clean:
	rm -f $(TARGET)
