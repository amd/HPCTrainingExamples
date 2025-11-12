all: openmp_code openmp_code1

AMDGPU_GFXMODEL = $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

CC1=$(notdir $(CC))

ifneq ($(findstring amdclang,$(CC1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=${AMDGPU_GFXMODEL}
else ifneq ($(findstring clang,$(CC1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=${AMDGPU_GFXMODEL}
else ifneq ($(findstring gcc,$(CC1)),)
  OPENMP_FLAGS = -fopenmp -foffload=-march=${AMDGPU_GFXMODEL}
else ifneq ($(findstring cc,$(CC1)),)
  OPENMP_FLAGS = -fopenmp
else
  OPENMP_FLAGS = -fopenmp --foffload=-march=${AMDGPU_GFXMODEL}
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
