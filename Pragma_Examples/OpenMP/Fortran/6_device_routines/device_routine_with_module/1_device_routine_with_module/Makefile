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
  FREE_FORM_FLAG = -Mfreeform
else ifneq ($(findstring gfortran,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload=-march=$(ROCM_GPU)
  FREE_FORM_FLAG = -ffree-form
else ifneq ($(findstring ftn,$(FC1)),)
  OPENMP_FLAGS = -fopenmp
  FTN_VERSION := $(shell $(FC) --version 2>&1)
  ifneq ($(findstring Cray,$(FTN_VERSION)),)
    FFLAGS = -g -eZ -O3 -f free ${OPENMP_FLAGS}
  else
    FFLAGS = -g -O3 -ffree-form ${OPENMP_FLAGS}
  endif
endif

LDFLAGS = ${OPENMP_FLAGS}

computemod.o: computemod.f90
	$(FC1) $(FFLAGS) $(LDFLAGS) -c computemod.f90

device_routine.o: device_routine.f90 computemod.o
	$(FC1) $(FFLAGS) $(LDFLAGS) -c device_routine.f90

device_routine: device_routine.o computemod.o
	$(FC1) $(FFLAGS) $(LDFLAGS)  -o device_routine device_routine.o computemod.o
# Cleanup
clean:
	rm -f *.o *.mod device_routine
