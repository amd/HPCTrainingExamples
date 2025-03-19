#This example was created by Johanna Potyka
# Copyright (c) 2024 AMD HPC Application Performance Team
# MIT License

default: device_routine
all: device_routine

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

FC1=$(notdir $(FC))

#ifeq ($(findstring amdflang,$(FC1)), amdflang)
#  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
#  FREE_FORM_FLAG = -Mfreeform
ifeq ($(findstring amdflang-new,$(FC1)), amdflang-new)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
  FREE_FORM_FLAG = -ffree-form
else ifeq ($(findstring flang,$(FC1)), flang)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
  FREE_FORM_FLAG = -Mfreeform
else ifeq ($(findstring gfortran,$(FC1)), gfortran)
  OPENMP_FLAGS = -fopenmp --offload=-march=$(ROCM_GPU)
  #OPENMP_FLAGS = -fopenmp -foffload=-march=${ROCM_GPU} -fopt-info-optimized-omp
  FREE_FORM_FLAG = -ffree-form
else ifeq ($(findstring ftn,$(FC1)), ftn)
  OPENMP_FLAGS = -fopenmp
  FREE_FORM_FLAG = -f free
endif

ifeq ($(findstring ftn,$(FC1)), ftn)
  FFLAGS = -g -eZ -O3 ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
else
  FFLAGS = -g -O3 ${FREE_FORM_FLAG} -fstrict-aliasing ${OPENMP_FLAGS}
endif
CFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS}
ifeq (${FC1},gfortran-13)
   LDFLAGS = ${OPENMP_FLAGS} -fno-lto
else ifeq ($(findstring ftn,$(FC1)), ftn)
   LDFLAGS = ${OPENMP_FLAGS} -L/opt/rh/gcc-toolset-12/root/usr/lib/gcc/x86_64-redhat-linux/12
else
   LDFLAGS = ${OPENMP_FLAGS}
endif


computemod.o: computemod.f90
	$(FC1) $(LDFLAGS) -c computemod.f90

device_routine.o: device_routine.f90 computemod.o
	$(FC1) $(LDFLAGS) -c device_routine.f90

device_routine: device_routine.o computemod.o
	$(FC1) $(LDFLAGS)  -o device_routine device_routine.o computemod.o
# Cleanup
clean:
	rm -f *.o *.mod device_routine
