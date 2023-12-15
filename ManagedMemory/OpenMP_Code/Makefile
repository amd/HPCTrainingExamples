all: openmp_code openmp_code1

CC1=$(notdir $(CC))

ifeq ($(findstring amdclang,$(CC1)), amdclang)
  OPENMP_FLAGS = -fopenmp --offload-arch=native
else ifeq ($(findstring clang,$(CC1)), clang)
  OPENMP_FLAGS = -fopenmp --offload-arch=native
else ifeq ($(findstring gcc,$(CC1)), gcc)
  OPENMP_FLAGS = -fopenmp -foffload=-march=native
else ifeq ($(findstring cc,$(CC1)), cc)
  OPENMP_FLAGS = -fopenmp
else
  OPENMP_FLAGS = -fopenmp --foffload=-march=native -fopt-info-optimized-omp
endif

CFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS}
LDFLAGS = ${OPENMP_FLAGS} -fno-lto -lm

openmp_code: openmp_code.o
	$(CC) $(LDFLAGS) $^ -o $@

openmp_code1: openmp_code1.o
	$(CC) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o openmp_code openmp_code1
