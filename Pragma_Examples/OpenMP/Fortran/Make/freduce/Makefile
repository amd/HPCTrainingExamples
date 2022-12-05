EXEC = freduce
default: ${EXEC}
all: ${EXEC}

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
UNAMEP = $(shell uname -p)
ROCM_CPUTARGET = $(UNAMEP)-pc-linux-gnu
ROCM_GPUTARGET ?= amdgcn-amd-amdhsa

ifeq ($(notdir $(FC)), flang)
# OPENMP_FLAGS = -fopenmp -target $(ROCM_CPUTARGET) -fopenmp-targets=$(ROCM_GPUTARGET) -Xopenmp-target=$(ROCM_GPUTARGET) -march=$(ROCM_GPU)
# Should work for > ROCm 4.5
  OPENMP_FLAGS = -fopenmp -target $(ROCM_CPUTARGET) --offload-arch=$(ROCM_GPU)
  FREE_FORM_FLAG = -Mfreeform
else ifeq ($(notdir $(FC)), amdflang)
# OPENMP_FLAGS = -fopenmp -target $(ROCM_CPUTARGET) -fopenmp-targets=$(ROCM_GPUTARGET) -Xopenmp-target=$(ROCM_GPUTARGET) -march=$(ROCM_GPU) 
# Should work for > ROCm 4.5
  OPENMP_FLAGS = -fopenmp -target $(ROCM_CPUTARGET) --offload-arch=$(ROCM_GPU)
  FREE_FORM_FLAG = -Mfreeform
else
  OPENMP_FLAGS = -fopenmp -foffload=-march=${ROCM_GPU} -fopt-info-optimized-omp
  FREE_FORM_FLAG = -ffree-form
endif

FFLAGS = -g -O3 ${FREE_FORM_FLAG} -fstrict-aliasing ${OPENMP_FLAGS}
LDFLAGS = ${OPENMP_FLAGS}

${EXEC}: ${EXEC}.o
	$(FC) $(LDFLAGS) $^ -o $@

clean:
	rm -f *.o ${EXEC} *.mod *.bin
