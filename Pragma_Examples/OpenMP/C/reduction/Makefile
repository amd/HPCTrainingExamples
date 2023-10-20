EXEC = reduction
default: ${EXEC}
all: ${EXEC}

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
UNAMEP = $(shell uname -p)
ROCM_CPUTARGET = $(UNAMEP)-pc-linux-gnu
ROCM_GPUTARGET ?= amdgcn-amd-amdhsa

CC1=$(notdir $(CC))

ifeq ($(findstring clang,$(CC1)), clang)
# OPENMP_FLAGS = -fopenmp -target $(ROCM_CPUTARGET) -fopenmp-targets=$(ROCM_GPUTARGET) -Xopenmp-target=$(ROCM_GPUTARGET) -march=$(ROCM_GPU) 
# Should work for > ROCm 4.5
  OPENMP_FLAGS = -fopenmp -target $(ROCM_CPUTARGET) --offload-arch=-march=$(ROCM_GPU)
else ifeq ($(findstring gcc,$(CC1)), gcc)
  OPENMP_FLAGS = -fopenmp -foffload=amdgcn-amdhsa=-march=$(ROCM_GPU)
else ifeq ($(findstring amdclang,$(CC1)), amdclang)
# OPENMP_FLAGS = -fopenmp -target $(ROCM_CPUTARGET) -fopenmp-targets=$(ROCM_GPUTARGET) -Xopenmp-target=$(ROCM_GPUTARGET) -march=$(ROCM_GPU) 
# Should work for > ROCm 4.5
  OPENMP_FLAGS = -fopenmp -target $(ROCM_CPUTARGET) --offload-arch=$(ROCM_GPU)
else ifeq ($(findstring cc,$(CC1)), cc)
  OPENMP_FLAGS = -fopenmp
else
  #OPENMP_FLAGS = --fopenmp -foffload=-march=${ROCM_GPU} -fopt-info-optimized-omp -fopt-info-missed
  OPENMP_FLAGS = --fopenmp -foffload=-march=${ROCM_GPU} -fopt-info-optimized-omp
endif

CFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS}
ifeq (${CC1},gcc-13)
   LDFLAGS = ${OPENMP_FLAGS} -fno-lto -lm
else
   LDFLAGS = ${OPENMP_FLAGS} -lm
endif

${EXEC}: ${EXEC}.o codelet.o
	$(CC) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o ${EXEC}
