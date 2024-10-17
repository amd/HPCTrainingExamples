# saxpy6 crashes during execution
all: saxpy


ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

CXX1=$(notdir $(CXX))

ifeq ($(findstring amdclang++,$(CXX1)), amdclang++)
  OPENMP_FLAGS = -fopenmp 
else ifeq ($(findstring clang++,$(CXX1)), clang++)
  OPENMP_FLAGS = -fopenmp 
else ifeq ($(findstring g++,$(CXX1)), g++)
  OPENMP_FLAGS = -fopenmp -foffload=-march=$(ROCM_GPU)
else ifeq ($(findstring CCi,$(CXX1)), CC) #carful: CC can be a compiler wrapper for many compilers!
  OPENMP_FLAGS = -fopenmp
else
  OPENMP_FLAGS = -fopenmp -fopt-info-optimized-omp
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
