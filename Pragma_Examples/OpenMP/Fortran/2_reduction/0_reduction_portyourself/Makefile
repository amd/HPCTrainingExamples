EXEC = freduce
default: ${EXEC}
all: ${EXEC}

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

FC1=$(notdir $(FC))

ifneq ($(findstring amdflang,$(FC1)),)
  OPENMP_FLAGS = 
  FREE_FORM_FLAG = -ffree-form
else ifneq ($(findstring flang,$(FC1)),)
  OPENMP_FLAGS = 
  FREE_FORM_FLAG = -ffree-form
else ifneq ($(findstring gfortran,$(FC1)),)
  OPENMP_FLAGS = 
  FREE_FORM_FLAG = -ffree-form
else ifneq ($(findstring ftn,$(FC1)),)
  CRAY_FORTRAN_COUNT=$(shell ftn --version | grep 'Cray Fortran' | wc -l)
  OPENMP_FLAGS =
  ifneq ("${CRAY_FORTRAN_COUNT}", 0)
     FREE_FORM_FLAG = -f free
     FFLAGS = -g -eZ -O3 ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
  else
     FREE_FORM_FLAG = -ffree-form
     FFLAGS = -g -O3 ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
  endif
endif

LDFLAGS = ${OPENMP_FLAGS}

${EXEC}: ${EXEC}.o
	$(FC) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o ${EXEC} *.mod *.bin
