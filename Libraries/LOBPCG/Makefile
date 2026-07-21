# Makefile for LOBPCG Eigenvalue Solver
# ROCm Implementation
#
# Compatible with ROCm 6.4+ and ROCm 7.x
#
# Usage:
#   module load rocm/7.1.1   # or rocm/6.4.0
#   make
#
# Targets:
#   lobpcg_mtx - LOBPCG eigenvalue solver with Matrix Market input

# ROCm installation path (set by module system or manually)
ROCM_PATH ?= /opt/rocm

# Compilers
HIPCC = $(ROCM_PATH)/bin/hipcc

# Compiler flags
HIPCC_FLAGS = -O3 -std=c++17 -munsafe-fp-atomics

# Include paths
INCLUDES = -I$(ROCM_PATH)/include -I.

# Library paths
LDFLAGS = -L$(ROCM_PATH)/lib

# Libraries
LIBS = -lrocsparse -lrocblas -lrocsolver -lrocrand -lamdhip64

# Target
TARGET = lobpcg_mtx

# Source files
DRIVER_SRC = driver_lobpcg.cpp
ALGO_SRC = lobpcg.cpp

# Object files
DRIVER_OBJ = driver_lobpcg.o
ALGO_OBJ = lobpcg.o

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(DRIVER_OBJ) $(ALGO_OBJ)
	$(HIPCC) $(HIPCC_FLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

$(DRIVER_OBJ): $(DRIVER_SRC) lobpcg.h
	$(HIPCC) $(HIPCC_FLAGS) $(INCLUDES) -c $< -o $@

$(ALGO_OBJ): $(ALGO_SRC) lobpcg.h
	$(HIPCC) $(HIPCC_FLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGET) $(DRIVER_OBJ) $(ALGO_OBJ)

# Run LOBPCG
# Example: make run MTX=matrix.mtx NEV=10 TOL=1e-6
run: $(TARGET)
	./$(TARGET) $(MTX) -nev $(NEV) -tol $(TOL)
