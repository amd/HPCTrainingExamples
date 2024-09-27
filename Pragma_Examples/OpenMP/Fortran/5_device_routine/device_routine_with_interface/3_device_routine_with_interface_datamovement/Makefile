#This example was created by Johanna Potyka
# Copyright (c) 2024 AMD HPC Application Performance Team
# MIT License

default: device_routine
all: device_routine

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

FC1=$(notdir $(FC))

ifeq ($(findstring amdflang,$(FC1)), amdflang)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
  FREE_FORM_FLAG = -Mfreeform
else ifeq ($(findstring flang-new,$(FC1)), flang-new)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
  FREE_FORM_FLAG = -ffree-form
else ifeq ($(findstring flang,$(FC1)), flang)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
  FREE_FORM_FLAG = -Mfreeform
else ifeq ($(findstring gfortran,$(FC1)), gfortran)
  OPENMP_FLAGS = -fopenmp --offload=-march=$(ROCM_GPU)
  FREE_FORM_FLAG = -ffree-form
else ifeq ($(findstring ftn,$(FC1)), ftn)
  OPENMP_FLAGS = -fopenmp
  FREE_FORM_FLAG = -f free
endif

ifeq ($(findstring ftn,$(FC1)), ftn)
  FFLAGS = -g -O3 ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
else
  FFLAGS = -g -O3 ${FREE_FORM_FLAG} -fstrict-aliasing ${OPENMP_FLAGS}
endif
CFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS}
ifeq (${FC1},gfortran-13)
   LDFLAGS = ${OPENMP_FLAGS} -fno-lto
else
   LDFLAGS = ${OPENMP_FLAGS}
endif

compute.o: compute.f90
	$(FC1) $(LDFLAGS) -c compute.f90

device_routine.o: device_routine.f90 compute.o
	$(FC1) $(LDFLAGS) -c device_routine.f90

device_routine: device_routine.o compute.o
	$(FC1) $(LDFLAGS)  -o device_routine device_routine.o compute.o
# Cleanup
clean:
	rm -f *.o *.mod device_routine
