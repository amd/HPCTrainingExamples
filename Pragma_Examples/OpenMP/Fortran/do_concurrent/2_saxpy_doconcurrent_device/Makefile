EXEC = saxpy
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

OPENMP_OFFLOAD_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}
DO_CONCURRENT_FLAGS = -fdo-concurrent-to-openmp=device

ifneq ($(findstring ftn,$(FC1)),)
  FFLAGS = -g -eZ -O3 ${FREE_FORM_FLAG}
else ifneq ($(findstring amdflang,$(FC1)),)
  FFLAGS = -g -O3 ${FREE_FORM_FLAG}
else
  FFLAGS = -g -O3 ${FREE_FORM_FLAG} -fstrict-aliasing
endif

${EXEC}.o: ${EXEC}.F90
	$(FC) -c $(FFLAGS) ${DO_CONCURRENT_FLAGS} ${OPENMP_OFFLOAD_FLAGS} $^

${EXEC}: ${EXEC}.o
	$(FC) ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

# Cleanup
clean:
	rm -f *.o ${EXEC} *.mod
