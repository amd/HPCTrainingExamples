# Makefile for PCG with Incomplete Cholesky Preconditioner
# ROCm Implementation
#
# Compatible with ROCm 6.4+ and ROCm 7.x
#
# Usage:
#   module load rocm/7.1.1   # or rocm/6.4.0
#   make
#
# Targets:
#   pcg_laplacian - PCG solver with 3D Laplacian test matrix
#   pcg_mtx       - PCG solver with Matrix Market input

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
LIBS = -lrocsparse -lrocblas -lamdhip64

# Targets
TARGET_LAPLACIAN = pcg_laplacian
TARGET_MTX = pcg_mtx

# Source files
DRIVER_LAPLACIAN_SRC = driver.cpp
DRIVER_MTX_SRC = driver_mtx.cpp
ALGO_SRC = pcg_ic.cpp

# Object files
DRIVER_LAPLACIAN_OBJ = driver.o
DRIVER_MTX_OBJ = driver_mtx.o
ALGO_OBJ = pcg_ic.o

.PHONY: all clean run run-mtx

all: $(TARGET_LAPLACIAN) $(TARGET_MTX)

$(TARGET_LAPLACIAN): $(DRIVER_LAPLACIAN_OBJ) $(ALGO_OBJ)
	$(HIPCC) $(HIPCC_FLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

$(TARGET_MTX): $(DRIVER_MTX_OBJ) $(ALGO_OBJ)
	$(HIPCC) $(HIPCC_FLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

$(DRIVER_LAPLACIAN_OBJ): $(DRIVER_LAPLACIAN_SRC) pcg_ic.h
	$(HIPCC) $(HIPCC_FLAGS) $(INCLUDES) -c $< -o $@

$(DRIVER_MTX_OBJ): $(DRIVER_MTX_SRC) pcg_ic.h
	$(HIPCC) $(HIPCC_FLAGS) $(INCLUDES) -c $< -o $@

$(ALGO_OBJ): $(ALGO_SRC) pcg_ic.h
	$(HIPCC) $(HIPCC_FLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGET_LAPLACIAN) $(TARGET_MTX) \
	      $(DRIVER_LAPLACIAN_OBJ) $(DRIVER_MTX_OBJ) $(ALGO_OBJ)

# Run Laplacian driver with default 20x20x20 grid
run: $(TARGET_LAPLACIAN)
	./$(TARGET_LAPLACIAN)

# Run Laplacian driver with custom grid size
# Example: make run-custom NX=50 NY=50 NZ=50 TOL=1e-10 MAXITER=500
run-custom: $(TARGET_LAPLACIAN)
	./$(TARGET_LAPLACIAN) $(NX) $(NY) $(NZ) $(TOL) $(MAXITER)

# Run Matrix Market driver
# Example: make run-mtx MTX=matrix.mtx TOL=1e-10 MAXITER=2000
run-mtx: $(TARGET_MTX)
	./$(TARGET_MTX) $(MTX) -tol $(TOL) -maxiter $(MAXITER)
