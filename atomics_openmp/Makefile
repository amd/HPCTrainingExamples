default: arraysum1 arraysum2 arraysum3 arraysum4 arraysum5 arraysum6 arraysum7 arraysum8 arraysum9 arraysum10
all: arraysum1 arraysum2 arraysum3 arraysum4 arraysum5 arraysum6 arraysum7 arraysum8 arraysum9 arraysum10

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
UNAMEP = $(shell uname -p)
ROCM_CPUTARGET = $(UNAMEP)-pc-linux-gnu
ROCM_GPUTARGET ?= amdgcn-amd-amdhsa

ifeq ($(notdir $(CC)), clang)
# OPENMP_FLAGS = -fopenmp -target $(ROCM_CPUTARGET) -fopenmp-targets=$(ROCM_GPUTARGET) -Xopenmp-target=$(ROCM_GPUTARGET) -march=$(ROCM_GPU) 
# Should work for > ROCm 4.5
  OPENMP_FLAGS = -fopenmp -target $(ROCM_CPUTARGET) --offload-arch=$(ROCM_GPU)
else ifeq ($(notdir $(CC)), amdclang)
# OPENMP_FLAGS = -fopenmp -target $(ROCM_CPUTARGET) -fopenmp-targets=$(ROCM_GPUTARGET) -Xopenmp-target=$(ROCM_GPUTARGET) -march=$(ROCM_GPU) 
# Should work for > ROCm 4.5
  OPENMP_FLAGS = -fopenmp -target $(ROCM_CPUTARGET) --offload-arch=$(ROCM_GPU)
else
  #OPENMP_FLAGS = -fopenmp -foffload=-march=${ROCM_GPU} -fopt-info-optimized-omp -fopt-info-missed
  OPENMP_FLAGS = -fopenmp -foffload=-march=${ROCM_GPU} -fopt-info-optimized-omp
endif

CFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS}
LDFLAGS = ${OPENMP_FLAGS} -lm

arraysum1: arraysum1.o
	$(CC) $(LDFLAGS) $^ -o $@

arraysum2: arraysum2.o
	$(CC) $(LDFLAGS) $^ -o $@

arraysum3: arraysum3.o
	$(CC) $(LDFLAGS) $^ -o $@

arraysum4: arraysum4.o
	$(CC) $(LDFLAGS) $^ -o $@

arraysum5: arraysum5.o
	$(CC) $(LDFLAGS) $^ -o $@

arraysum6: arraysum6.o
	$(CC) $(LDFLAGS) $^ -o $@

arraysum7: arraysum7.o
	$(CC) $(LDFLAGS) $^ -o $@

arraysum8: arraysum8.o
	$(CC) $(LDFLAGS) $^ -o $@

arraysum9: arraysum9.o
	$(CC) $(LDFLAGS) $^ -o $@

arraysum10: arraysum10.o
	$(CC) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o arraysum1 arraysum2 arraysum3 arraysum4 arraysum5 arraysum6 arraysum7 arraysum8 arraysum9 arraysum10
