##**************************************************************************
##* Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
##**************************************************************************

# Compilers and related
ROCM_PATH     ?= /opt/rocm
ROCM_GPU      ?= $(INSTALLED_GPU)
INSTALLED_GPU = $(shell $(ROCM_PATH)/bin/rocm_agent_enumerator | grep -m 1 -E gfx[^0]{1})
CFLAG         = -Ofast -g --offload-arch=$(ROCM_GPU)
LDFLAG        =

# OpenMP
COMPILER_OMP  = $(ROCM_PATH)/llvm/bin/clang++
CFLAGS_OMP    = -fopenmp
LDFLAGS_OMP   =

# HIP
COMPILER_HIP  = $(ROCM_PATH)/bin/hipcc
CFLAGS_HIP    = -D_HIP
LDFLAGS_HIP   =


# Source code
OBJS= Jacobi.o \
  Laplacian.o \
	BoundaryConditions.o \
  Update.o \
	Norm.o \
	Input.o \
	Main.o

# Commands
VERSIONS=omp hip
all: $(VERSIONS)

omp:
	@echo "======================="
	@echo "Building OpenMP version"
	@echo "======================="
	mkdir -p build & mkdir -p build/omp
	rsync -ru *.cpp *.hpp Makefile build/omp
	$(MAKE) -C build/$@ jacobi CC=$(COMPILER_OMP) CFLAGS=$(CFLAGS_OMP) LDFLAGS=$(LDFLAGS_OMP)
	cp build/omp/Jacobi ./Jacobi_omp

hip:
	@echo "===================="
	@echo "Building HIP version"
	@echo "===================="
	mkdir -p build & mkdir -p build/hip
	rsync -ru *.cpp *.hpp Makefile build/hip
	$(MAKE) -C build/$@ jacobi CC=$(COMPILER_HIP) CFLAGS=$(CFLAGS_HIP) LDFLAGS=$(LDFLAGS_HIP)
	cp build/hip/Jacobi ./Jacobi_hip

jacobi: $(OBJS)
	$(CC) $(CFLAG) $(CFLAGS) $(LDFLAG) $(LDFLAGS) -o Jacobi  $(OBJS)

%.o : %.cpp
	$(CC) $(CFLAG) $(CFLAGS) -c $< -o $@

clean_omp:
	rm -rf ./Jacobi_omp build/omp

clean_hip:
	rm -rf ./Jacobi_hip build/hip

clean: clean_omp clean_hip
	rm -rf build/
