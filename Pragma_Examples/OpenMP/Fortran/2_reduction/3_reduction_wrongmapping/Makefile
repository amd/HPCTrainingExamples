EXEC = freduce
default: ${EXEC}
all: ${EXEC}

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

FC1=$(notdir $(FC))

ifneq ($(findstring amdflang,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
  FFLAGS = -g -O3 -ffree-form ${OPENMP_FLAGS}
else ifneq ($(findstring flang,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
  FFLAGS = -g -O3 -ffree-form ${OPENMP_FLAGS}
else ifneq ($(findstring gfortran,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload=-march=$(ROCM_GPU)
  FFLAGS = -g -O3 -ffree-form ${OPENMP_FLAGS}
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

${EXEC}: ${EXEC}.o
	$(FC) $(FFLAGS) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o ${EXEC} *.mod *.bin
