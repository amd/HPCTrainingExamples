#This example was created by Johanna Potyka
# Copyright (c) 2024 AMD HPC Application Performance Team
# MIT License

default: device_routine
all: device_routine

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

FC1=$(notdir $(FC))

ifneq ($(findstring amdflang,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
  FREE_FORM_FLAG = -ffree-form
else ifneq ($(findstring flang,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
  FREE_FORM_FLAG = -ffree-form
else ifneq ($(findstring gfortran,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload=-march=$(ROCM_GPU)
  #OPENMP_FLAGS = -fopenmp -foffload=-march=${ROCM_GPU} -fopt-info-optimized-omp
  FREE_FORM_FLAG = -ffree-form
else ifneq ($(findstring ftn,$(FC1)),)
  FTN_VERSION := $(shell $(FC) --version 2>&1)
  OPENMP_FLAGS = -fopenmp
  ifneq ($(findstring Cray,$(FTN_VERSION)),)
    FREE_FORM_FLAG = -f free
    FFLAGS = -eZ
  else
    FREE_FORM_FLAG = -ffree-form
    FFLAGS =
  endif
endif

FFLAGS := -g -O3 ${FFLAGS} ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
LDFLAGS = ${OPENMP_FLAGS}

computemod.o: computemod.f90
	$(FC) $(FFLAGS) $(LDFLAGS) -c computemod.f90

device_routine.o: device_routine.f90 computemod.o
	$(FC) $(FFLAGS) $(LDFLAGS) -c device_routine.f90

device_routine: device_routine.o computemod.o
	$(FC) $(LDFLAGS)  -o device_routine device_routine.o computemod.o

# Cleanup
clean:
	rm -f *.o *.mod device_routine
