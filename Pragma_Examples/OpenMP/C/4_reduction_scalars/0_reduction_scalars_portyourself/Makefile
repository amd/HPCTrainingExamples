EXEC = reduction_scalar
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
  OPENMP_FLAGS = -fopenmp 
else ifneq ($(findstring clang,$(CC1)),)
  OPENMP_FLAGS = -fopenmp 
else ifneq ($(findstring gcc,$(CC1)),)
  OPENMP_FLAGS = -fopenmp 
else ifneq ($(findstring cc,$(CC1)),)
  OPENMP_FLAGS = -fopenmp
else
  OPENMP_FLAGS = -fopenmp -fopt-info-optimized-omp
endif

CFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS}
ifeq (${CC1},gcc-13)
   LDFLAGS = ${OPENMP_FLAGS} -fno-lto -lm
else
   LDFLAGS = ${OPENMP_FLAGS} -lm
endif

${EXEC}: ${EXEC}.o
	$(COMPILER) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o ${EXEC}
