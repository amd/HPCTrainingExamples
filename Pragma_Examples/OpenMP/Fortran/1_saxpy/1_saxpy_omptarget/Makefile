EXEC = saxpy
default: ${EXEC}
all: ${EXEC}

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

FC1=$(notdir $(FC))

ifneq ($(findstring amdflang,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
  FREE_FORM_FLAG = -ffree-form
  FFLAGS = -g -O3 ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
else ifneq ($(findstring flang,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
  FREE_FORM_FLAG = -Mfreeform
  FFLAGS = -g -O3 ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
else ifneq ($(findstring gfortran,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload=-march=$(ROCM_GPU)
  #OPENMP_FLAGS = -fopenmp -foffload=-march=${ROCM_GPU} -fopt-info-optimized-omp
  FREE_FORM_FLAG = -ffree-form
else ifneq ($(findstring ftn,$(FC1)),)
  FTN_VERSION := $(shell $(FC) --version 2>&1)
  ifneq ($(findstring Cray,$(FTN_VERSION)),)
    OPENMP_FLAGS = -fopenmp
    FFLAGS = -g -eZ -O3 ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
  else
    OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
    FFLAGS = -g -O3 -ffree-form ${OPENMP_FLAGS}
  endif
endif

LDFLAGS = ${OPENMP_FLAGS}


${EXEC}: ${EXEC}.F90
	$(FC) $(FFLAGS) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o ${EXEC} *.mod
