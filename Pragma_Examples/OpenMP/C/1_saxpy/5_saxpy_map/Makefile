# saxpy6 crashes during execution
all: saxpy


ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

CXX1=$(notdir $(CXX))

ifneq ($(findstring amdclang++,$(CXX1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
else ifneq ($(findstring clang++,$(CXX1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
else ifneq ($(findstring g++,$(CXX1)),)
  OPENMP_FLAGS = -fopenmp -foffload=-march=$(ROCM_GPU)
else ifneq ($(findstring CC,$(CXX1)),) #careful: CC can be a compiler wrapper for many compilers on cray systems!
  OPENMP_FLAGS = -fopenmp
else
  OPENMP_FLAGS = -fopenmp --foffload=-march=$(ROCM_GPU) -fopt-info-optimized-omp
endif

CFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS}

ifeq (${CXX1},g++-13)
   LDFLAGS = ${OPENMP_FLAGS} -fno-lto -lm
else
   LDFLAGS = ${OPENMP_FLAGS} -lm
endif

saxpy.o: saxpy.cpp
	$(CXX) $(CFLAGS) -c saxpy.cpp  -o saxpy.o

saxpy: saxpy.o
	$(CXX)  $(LDFLAGS) $^ -o $@

run: saxpy
	./saxpy

clean:
	rm -f saxpy *.o
