EXEC = dtype_derived_type dtype_pointer dtype_scalar_members dtype_derived_type_usm dtype_derived_type_automap
default: ${EXEC}
all: ${EXEC}

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

dtype_derived_type_automap: dtype_derived_type_automap.o
	$(FC) $(LDFLAGS) $^ -o $@

dtype_derived_type: dtype_derived_type.o
	$(FC) $(LDFLAGS) $^ -o $@

dtype_derived_type_usm: dtype_derived_type_usm.o
	$(FC) $(LDFLAGS) $^ -o $@

dtype_pointer: dtype_pointer.o
	$(FC) $(LDFLAGS) $^ -o $@

dtype_scalar_members: dtype_scalar_members.o
	$(FC) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o ${EXEC} *.mod pgipar*
