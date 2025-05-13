all: memorypool

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

FC1=$(notdir $(FC))

ifneq ($(findstring amdflang, $(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}
  FREE_FORM_FLAG = -ffree-form
else ifneq ($(findstring flang, $(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}
  FREE_FORM_FLAG = -ffree-form
else ifneq ($(findstring gfortran,$(FC1)),)
  OPENMP_FLAGS = -fopenmp -foffload=-march=${ROCM_GPU}
  FREE_FORM_FLAG = -ffree-form
else ifneq ($(findstring ftn,$(FC1)),)
  OPENMP_FLAGS = -fopenmp
endif

FFLAGS = -g -O3 ${FREE_FORM_FLAG} ${OPENMP_FLAGS} -I${UMPIRE_PATH}/include/umpire
ifeq (${FC1},gfortran-13)
  LDFLAGS = ${OPENMP_FLAGS} -fno-lto
else
  LDFLAGS = ${OPENMP_FLAGS} -L${UMPIRE_PATH}/lib64 -L${UMPIRE_PATH}/lib -lumpire -lcamp -lfmt -L${ROCM_PATH}/lib -lamdhip64 -lstdc++
endif

memorypool.o: memorypool.f90
	$(FC) -c $(FFLAGS) $^

memorypool: memorypool.o
	$(FC) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o memorypool *.mod
	rm -rf build
