EXEC = declare_target_global
default: ${EXEC}
all: ${EXEC}

# Pick the C compiler from C_COMPILER, then CC, otherwise fall back to amdclang.
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
  OPENMP_FLAGS = -fopenmp --foffload=-march=$(ROCM_GPU)
endif

CFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS} -fopenmp-force-usm

ifeq (${CC1},gcc-13)
  LDFLAGS = ${OPENMP_FLAGS} -fno-lto -lm
else
  LDFLAGS = ${OPENMP_FLAGS} -lm
endif

${EXEC}: ${EXEC}.c
	$(COMPILER) $(CFLAGS) $^ -o $@ $(LDFLAGS)

run: ${EXEC}
	./${EXEC}

clean:
	rm -f ${EXEC} *.o
