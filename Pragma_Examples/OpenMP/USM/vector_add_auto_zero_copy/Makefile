EXEC = vector_add_auto_zero_copy
default: ${EXEC}
all: ${EXEC}

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

CXX1=$(notdir $(CXX))

ifneq ($(findstring amdclang,$(CXX1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
else ifneq ($(findstring clang,$(CXX1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
else ifneq ($(findstring gcc,$(CXX1)),)
  OPENMP_FLAGS = -fopenmp -foffload=-march=$(ROCM_GPU)
else ifneq ($(findstring CC,$(CXX1)),)
  OPENMP_FLAGS = -fopenmp
else
  OPENMP_FLAGS = -fopenmp --foffload=-march=$(ROCM_GPU) -fopt-info-optimized-omp
endif

CXXFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS}
LDFLAGS = ${OPENMP_FLAGS} -fno-lto -lm

${EXEC}: ${EXEC}.o
	$(CXX) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o ${EXEC}

HSA_XNACK ?= 1
OMPX_APU_MAPS ?= 1

run:
	OMPX_APU_MAPS=${OMPX_APU_MAPS} HSA_XNACK=${HSA_XNACK} ./$(EXEC)
