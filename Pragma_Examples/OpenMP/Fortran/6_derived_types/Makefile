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
  FREE_FORM_FLAG = -Mfreeform
else ifneq ($(findstring gfortran,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload=-march=$(ROCM_GPU)
  FREE_FORM_FLAG = -ffree-form
else ifneq ($(findstring ftn,$(FC1)),)
  OPENMP_FLAGS = -fopenmp
endif

ifneq ($(findstring ftn,$(FC1)),)
  FFLAGS = -g -eZ -O3 ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
else ifneq ($(findstring amdflang,$(FC1)),)
  FFLAGS = -g -O3 ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
else
  FFLAGS = -g -O3 ${FREE_FORM_FLAG} -fstrict-aliasing ${OPENMP_FLAGS}
endif
ifeq (${FC1},gfortran-13)
   LDFLAGS = ${OPENMP_FLAGS} -fno-lto
else ifeq ($(findstring ftn,$(FC1)), ftn)
   LDFLAGS = ${OPENMP_FLAGS} -L/opt/rh/gcc-toolset-12/root/usr/lib/gcc/x86_64-redhat-linux/12
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
