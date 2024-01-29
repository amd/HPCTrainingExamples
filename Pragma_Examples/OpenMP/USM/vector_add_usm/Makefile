EXEC = vector_add_usm
default: ${EXEC}
all: ${EXEC}

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

CXX1=$(notdir $(CXX))

ifeq ($(findstring amdclang,$(CXX1)), amdclang)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
else ifeq ($(findstring clang,$(CXX1)), clang)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
else ifeq ($(findstring gcc,$(CXX1)), gcc)
  OPENMP_FLAGS = -fopenmp -foffload=-march=$(ROCM_GPU)
else ifeq ($(findstring cc,$(CXX1)), cc)
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

run:
	HSA_XNACK=${HSA_XNACK} ./$(EXEC)
