##**************************************************************************
##* Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
##**************************************************************************

# Compilers and related
ROCM_PATH     ?= /opt/rocm
ROCM_GPU      ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
CXXFLAG       = -O2 -g --offload-arch=$(ROCM_GPU)
LDFLAG        =

# OpenMP
COMPILER_OMP  = $(CXX) # e.g. $(ROCM_PATH)/llvm/bin/clang++ here: accept from environment
CXXFLAGS_OMP  = '-fopenmp --offload-arch=${ROCM_GPU}'  #if offload-arch is needed this depends on the compiler chosen!
LDFLAGS_OMP   =


# Source code
OBJS=	Jacobi.o \
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
	$(MAKE) -C build/$@ jacobi CXX=$(COMPILER_OMP) CXXFLAGS=$(CXXFLAGS_OMP) LDFLAGS=$(LDFLAGS_OMP)
	cp build/omp/Jacobi ./Jacobi_omp


jacobi: $(OBJS)
	$(CXX) $(CXXFLAG) $(CXXFLAGS) $(LDFLAG) $(LDFLAGS) -o Jacobi  $(OBJS)

%.o : %.cpp
	$(CXX) $(CXXFLAG) $(CXXFLAGS) -c $< -o $@

clean_omp:
	rm -rf ./Jacobi_omp build/omp

clean: clean_omp
	rm -rf build/
