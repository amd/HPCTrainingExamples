all:saxpy_cpu saxpy_gpu_singleunit_autoalloc saxpy_singleunit_dynamic saxpy_gpu_paralleldo saxpy_gpu_loop saxpy_gpu_do_concurrent_host saxpy_gpu_do_concurrent_device saxpy_gpu_collapse

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

OPENMP_FLAGS = -fopenmp
OPENMP_OFFLOAD_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}

FFLAGS = -g -O3 ${FREE_FORM_FLAG}

saxpy_cpu.o: saxpy_cpu.F90
	$(FC) -c $(FFLAGS) ${OPENMP_FLAGS} $^

saxpy_cpu: saxpy_cpu.o
	$(FC) $(LDFLAGS) ${OPENMP_FLAGS} $^ -o $@

saxpy_gpu_singleunit_autoalloc.o: saxpy_gpu_singleunit_autoalloc.F90
	$(FC) -c $(FFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^

saxpy_gpu_singleunit_autoalloc: saxpy_gpu_singleunit_autoalloc.o
	$(FC) $(LDFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

saxpy_gpu_singleunit_dynamic.o: saxpy_gpu_singleunit_dynamic.F90
	$(FC) -c $(FFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^

saxpy_gpu_singleunit_dynamic: saxpy_gpu_singleunit_dynamic.o
	$(FC) $(LDFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

saxpy_gpu_paralleldo.o: saxpy_gpu_paralleldo.F90
	$(FC) -c $(FFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^

saxpy_gpu_paralleldo: saxpy_gpu_paralleldo.o
	$(FC) $(LDFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

saxpy_gpu_loop.o: saxpy_gpu_loop.F90
	$(FC) -c $(FFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^

saxpy_gpu_loop: saxpy_gpu_loop.o
	$(FC) $(LDFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

saxpy_gpu_do_concurrent_host.o: saxpy_gpu_do_concurrent_host.F90
	#@echo "not implemented yet in amdflang next gen compiler"
	$(FC) -c $(FFLAGS) -fdo-concurrent-parallel=host ${OPENMP_OFFLOAD_FLAGS} $^

saxpy_gpu_do_concurrent_host: saxpy_gpu_do_concurrent_host.o
	$(FC) $(LDFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

saxpy_gpu_do_concurrent_device.o: saxpy_gpu_do_concurrent_device.F90
	#@echo "not implemented yet in amdflang next gen compiler"
	$(FC) -c $(FFLAGS) -fdo-concurrent-parallel=device ${OPENMP_OFFLOAD_FLAGS} $^

saxpy_gpu_do_concurrent_device: saxpy_gpu_do_concurrent_device.o
	$(FC) $(LDFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

saxpy_gpu_collapse.o: saxpy_gpu_collapse.F90
	$(FC) -c $(FFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^

saxpy_gpu_collapse: saxpy_gpu_collapse.o
	$(FC) $(LDFLAGS) ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

# Cleanup
clean:
	rm -f *.o saxpy_cpu saxpy_gpu_singleunit_autoalloc saxpy_singleunit_dynamic saxpy_gpu_paralleldo saxpy_gpu_loop saxpy_gpu_do_concurrent_host saxpy_gpu_do_concurrent_device saxpy_gpu_collapse *.mod
