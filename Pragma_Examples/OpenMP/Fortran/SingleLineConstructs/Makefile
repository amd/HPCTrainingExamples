all:saxpy_cpu

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

FC1=$(notdir $(FC))

ifeq ($(findstring amdflang-new, $(FC1)), amdflang-new)
  OPENMP_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}
  FREE_FORM_FLAG = -ffree-form
else ifeq ($(findstring flang, $(FC1)), flang)
  OPENMP_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}
  FREE_FORM_FLAG = -Mfreeform
else ifeq ($(findstring amdflang, $(FC1)), amdflang)
  OPENMP_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}
  FREE_FORM_FLAG = -Mfreeform
else
  OPENMP_FLAGS = -fopenmp -foffload=-march=${ROCM_GPU}
  FREE_FORM_FLAG = -ffree-form
endif

FFLAGS = -g -O3 ${FREE_FORM_FLAG} -fopenmp
LDFLAGS = -fopenmp

saxpy_cpu.o: saxpy_cpu.F90
	$(FC) -c $(FFLAGS) $^

saxpy_cpu: saxpy_cpu.o
	$(FC) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o saxpy_cpu *.mod
	rm -rf build
