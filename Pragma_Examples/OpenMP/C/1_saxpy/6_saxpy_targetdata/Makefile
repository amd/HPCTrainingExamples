# saxpy6 crashes during execution
all: saxpy


ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

CC1=$(notdir $(CC))

ifneq ($(findstring amdclang,$(CC1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
else ifneq ($(findstring clang,$(CC1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
else ifneq ($(findstring gcc,$(CC1)),)
  OPENMP_FLAGS = -fopenmp -foffload=-march=$(ROCM_GPU)
else ifneq ($(findstring cc,$(CC1)),) #careful: CC can be a compiler wrapper for many compilers on cray systems!
  OPENMP_FLAGS = -fopenmp
else
  OPENMP_FLAGS = -fopenmp --foffload=-march=$(ROCM_GPU) -fopt-info-optimized-omp
endif

CFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS}

ifeq (${CC1},gcc-13)
   LDFLAGS = ${OPENMP_FLAGS} -fno-lto -lm
else
   LDFLAGS = ${OPENMP_FLAGS} -lm
endif

saxpy.o: saxpy.c
	$(CC) $(CFLAGS) -c saxpy.c  -o saxpy.o

saxpy: saxpy.o
	$(CC)  $(LDFLAGS) $^ -o $@

run: saxpy
	./saxpy

clean:
	rm -f saxpy *.o
