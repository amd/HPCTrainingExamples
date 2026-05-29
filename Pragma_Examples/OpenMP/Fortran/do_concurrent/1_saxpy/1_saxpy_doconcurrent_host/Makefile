EXEC = saxpy
default: ${EXEC}
all: ${EXEC}

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

FC1=$(notdir $(FC))

FREE_FORM_FLAG = -ffree-form

OPENMP_OFFLOAD_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}
DO_CONCURRENT_FLAGS = -fdo-concurrent-to-openmp=host

FFLAGS = -g -O3 ${FREE_FORM_FLAG}

${EXEC}.o: ${EXEC}.F08
	$(FC) -c $(FFLAGS) ${DO_CONCURRENT_FLAGS} ${OPENMP_FLAGS} $^

${EXEC}: ${EXEC}.o
	$(FC) ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

# Cleanup
clean:
	rm -f *.o ${EXEC} *.mod
