all:saxpy_cpu saxpy_gpu_collapse

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

OPENMP_FLAGS = -fopenmp
OPENMP_OFFLOAD_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}

FFLAGS = -g -O3 ${FREE_FORM_FLAG}

saxpy_cpu.o: saxpy_cpu.F90
	$(FC) -c $(FFLAGS) ${OPENMP_FLAGS} $^

saxpy_cpu: saxpy_cpu.o
	$(FC) $(LDFLAGS) ${OPENMP_FLAGS} $^ -o $@

saxpy_gpu_collapse.o: saxpy_gpu_collapse.F90
	$(FC) -c $(FFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^

saxpy_gpu_collapse: saxpy_gpu_collapse.o
	$(FC) $(LDFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

# Cleanup
clean:
	rm -f *.o saxpy_cpu saxpy_gpu_collapse *.mod
