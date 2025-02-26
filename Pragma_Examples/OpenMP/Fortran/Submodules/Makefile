all:impl.o interface.o

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

OPENMP_FLAGS = -fopenmp
OPENMP_OFFLOAD_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}

FREE_FORM_FLAG = -ffree-form
FFLAGS = -g -O3 ${FREE_FORM_FLAG}

impl.o: impl.f90
	$(FC) -c $(FFLAGS) ${OPENMP_FLAGS} $<

interface.o: interface.f90
	$(FC) -c $(FFLAGS) ${OPENMP_FLAGS} $<

impl.o: interface.o

# Cleanup
clean:
	rm -f *.o *.mod impl interface
