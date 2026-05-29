EXEC = common_blocks
default: ${EXEC}
all: ${EXEC}

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

FC1=$(notdir $(FC))

ifneq ($(findstring amdflang,$(FC1)),)
  OPENMP_FLAGS = -fopenmp
  FREE_FORM_FLAG = -ffree-form
else ifneq ($(findstring flang,$(FC1)),)
  OPENMP_FLAGS = -fopenmp
  FREE_FORM_FLAG = -Mfreeform
else ifneq ($(findstring gfortran,$(FC1)),)
  OPENMP_FLAGS = -fopenmp
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
else ifneq ($(findstring ftn,$(FC1)),)
   LDFLAGS = ${OPENMP_FLAGS}
else
   LDFLAGS = ${OPENMP_FLAGS}
endif

ifneq ($(findstring ftn,$(FC1)),)
  OFFLOAD_FLAGS =
else
  OFFLOAD_FLAGS = --offload-arch=${ROCM_GPU}
endif

${EXEC}: ${EXEC}.f90
	$(FC) $(FFLAGS) $(OFFLOAD_FLAGS) $(LDFLAGS) $^ -o $@

clean:
	rm -f *.o ${EXEC} *.mod
