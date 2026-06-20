EXEC = atomic
default: ${EXEC}
all: ${EXEC}

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
     FREE_FORM_FLAG = -f free
     FFLAGS = -g -eZ -O3 ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
  else
     FREE_FORM_FLAG = -ffree-form
     FFLAGS = -g -O3 ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
  endif
endif

LDFLAGS = ${OPENMP_FLAGS}

${EXEC}: ${EXEC}.o
	$(FC) $(FFLAGS) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o ${EXEC} *.mod *.bin
