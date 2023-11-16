EXEC = reduction
default: ${EXEC}
all: ${EXEC}

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

CC1=$(notdir $(CC))

ifeq ($(findstring clang, $(CC1)), clang)
  OPENACC_FLAGS = -fopenacc --offload-arch=${ROCM_GPU}
else ifeq ($(findstring amdclang, $(CC1)), amdclang)
  OPENACC_FLAGS = -fopenacc --offload-arch=${ROCM_GPU}
else
  OPENACC_FLAGS = -fopenacc -foffload=-march=${ROCM_GPU} -fopt-info-optimized-omp
endif

CFLAGS = -g -O3 -fstrict-aliasing ${OPENACC_FLAGS}
ifeq (${CC1},gcc-13)
  LDFLAGS = ${OPENACC_FLAGS} -fno-lto -lm
else
  LDFLAGS = ${OPENACC_FLAGS} -lm
endif

${EXEC}: ${EXEC}.o codelet.o
	$(CC) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o ${EXEC}
