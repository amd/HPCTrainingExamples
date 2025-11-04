EXEC = reduction_array
default: ${EXEC}
all: ${EXEC}


# To avoid confusion with CC/cc Cray compiler wrappers we follow this procedure
# 1. Take compiler specified in C_COMPILER environment variable if set. If not,
# 2. Use the standard CC variable to idenetify compiler.
# 3. As last resort, use the amdclang compiler as default
COMPILER := $(or $(C_COMPILER),$(CC),amdclang)

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

CC1=$(notdir $(COMPILER))

ifneq ($(findstring amdclang,$(CC1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
else ifneq ($(findstring clang,$(CC1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
else ifneq ($(findstring gcc,$(CC1)),)
  OPENMP_FLAGS = -fopenmp -foffload=-march=$(ROCM_GPU)
else ifneq ($(findstring cc,$(CC1)),)
  OPENMP_FLAGS = -fopenmp
else
  #OPENMP_FLAGS = -fopenmp --foffload=-march=$(ROCM_GPU) -fopt-info-optimized-omp -fopt-info-missed
  OPENMP_FLAGS = -fopenmp --foffload=-march=$(ROCM_GPU) -fopt-info-optimized-omp
endif

CFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS}
LDFLAGS = ${OPENMP_FLAGS} -fno-lto -lm

${EXEC}.o: ${EXEC}.c
	$(COMPILER) $(CFLAGS) -c ${EXEC}.c -o ${EXEC}.o

${EXEC}: ${EXEC}.o
	$(COMPILER) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o ${EXEC}
