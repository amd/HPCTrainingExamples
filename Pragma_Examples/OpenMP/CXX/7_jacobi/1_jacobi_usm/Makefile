##**************************************************************************
##* Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
##**************************************************************************

# Compilers and related
ROCM_PATH     ?= /opt/rocm
ROCM_GPU      ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//')) 
CFLAG         = -Ofast -g --offload-arch=$(ROCM_GPU)
LDFLAG        =

# OpenMP
COMPILER_OMP  = $(CC) # e.g. $(ROCM_PATH)/llvm/bin/clang++ here: accept from environment
CFLAGS_OMP    = '-fopenmp --offload-arch=${ROCM_GPU}'  #if offload-arch is needed this depends on the compiler chosen!
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
VERSIONS=omp
all: $(VERSIONS)

omp:
	@echo "======================="
	@echo "Building OpenMP version"
	@echo "======================="
	mkdir -p build & mkdir -p build/omp
	rsync -ru *.cpp *.hpp Makefile build/omp
	$(MAKE) -C build/$@ jacobi CC=$(COMPILER_OMP) CFLAGS=$(CFLAGS_OMP) LDFLAGS=$(LDFLAGS_OMP)
	cp build/omp/Jacobi ./Jacobi_omp


jacobi: $(OBJS)
	$(CC) $(CFLAG) $(CFLAGS) $(LDFLAG) $(LDFLAGS) -o Jacobi  $(OBJS)

%.o : %.cpp
	$(CC) $(CFLAG) $(CFLAGS) -c $< -o $@

clean_omp:
	rm -rf ./Jacobi_omp build/omp


clean: clean_omp
	rm -rf build/
