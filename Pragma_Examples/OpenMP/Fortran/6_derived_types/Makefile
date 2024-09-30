EXEC = dtype_derived_type  dtype_pointer dtype_scalar_members dtype_derived_type_usm dtype_derived_type_automap
default: ${EXEC}
all: ${EXEC}

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
  FREE_FORM_FLAG = -ffree-form
else ifeq ($(findstring ftn,$(FC1)), ftn)
  OPENMP_FLAGS = -fopenmp
endif

ifeq ($(findstring ftn,$(FC1)), ftn)
  FFLAGS = -g -O3 ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
else ifeq ($(findstring flang-new,$(FC1)), flang-new)
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
