#This example was created by Johanna Potyka
# Copyright (c) 2024 AMD HPC Application Performance Team
# MIT License

default: device_routine
all: device_routine

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

FC1=$(notdir $(FC))

ifneq ($(findstring amdflang,$(FC1)),)
  OPENMP_FLAGS = 
  FREE_FORM_FLAG = -ffree-form
else ifneq ($(findstring flang,$(FC1)),)
  OPENMP_FLAGS = 
  FREE_FORM_FLAG = -Mfreeform
else ifneq ($(findstring gfortran,$(FC1)),)
  OPENMP_FLAGS = 
  FREE_FORM_FLAG = -ffree-form
else ifneq ($(findstring ftn,$(FC1)),)
  OPENMP_FLAGS = 
  FREE_FORM_FLAG = -f free
endif

ifneq ($(findstring ftn,$(FC1)),)
  FFLAGS = -g -eZ -O3 ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
else
  FFLAGS = -g -O3 ${FREE_FORM_FLAG} -fstrict-aliasing ${OPENMP_FLAGS}
endif
ifeq (${FC1},gfortran-13)
   LDFLAGS = ${OPENMP_FLAGS} -fno-lto
else ifneq ($(findstring ftn,$(FC1)),)
   LDFLAGS = ${OPENMP_FLAGS} -L/opt/rh/gcc-toolset-12/root/usr/lib/gcc/x86_64-redhat-linux/12
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
