EXEC = saxpy
default: ${EXEC}
all: ${EXEC}

CC1=$(notdir $(CC))

ifneq ($(findstring clang, $(CC1)),)
  OPENACC_FLAGS = -fopenacc --offload-arch=native
else ifneq ($(findstring amdclang, $(CC1)),)
  OPENACC_FLAGS = -fopenacc --offload-arch=native
else
  OPENACC_FLAGS = -fopenacc -foffload=-march=native -fopt-info-optimized-omp
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
