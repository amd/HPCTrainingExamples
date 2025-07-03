all: saxpy_cpu saxpy_gpu_singleunit_autoalloc saxpy_gpu_singleunit_dynamic saxpy_gpu_parallelfor saxpy_gpu_loop saxpy_gpu_collapse

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

OPENMP_FLAGS=-fopenmp
OPENMP_OFFLOAD_FLAGS=-fopenmp --offload-arch=${ROCM_GPU}

CFLAGS = -g -O3 -fstrict-aliasing

LDFLAGS = -lm

saxpy_cpu.o: saxpy_cpu.c
	$(CC) $(CFLAGS) ${OPENMP_FLAGS} -c $<  -o $@

saxpy_cpu: saxpy_cpu.o
	$(CC)  $(LDFLAGS) ${OPENMP_FLAGS} $^ -o $@

saxpy_gpu_singleunit_autoalloc.o: saxpy_gpu_singleunit_autoalloc.c
	$(CC) $(CFLAGS) ${OPENMP_OFFLOAD_FLAGS} -c $<  -o $@

saxpy_gpu_singleunit_autoalloc: saxpy_gpu_singleunit_autoalloc.o
	$(CC)  $(LDFLAGS)  ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

saxpy_gpu_singleunit_dynamic.o: saxpy_gpu_singleunit_dynamic.c
	$(CC) $(CFLAGS) ${OPENMP_OFFLOAD_FLAGS} -c $<  -o $@

saxpy_gpu_singleunit_dynamic: saxpy_gpu_singleunit_dynamic.o
	$(CC)  $(LDFLAGS)  ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

saxpy_gpu_parallelfor.o: saxpy_gpu_parallelfor.c
	$(CC) $(CFLAGS) ${OPENMP_OFFLOAD_FLAGS} -c $<  -o $@

saxpy_gpu_parallelfor: saxpy_gpu_parallelfor.o
	$(CC)  $(LDFLAGS)  ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

saxpy_gpu_loop.o: saxpy_gpu_loop.c
	$(CC) $(CFLAGS) ${OPENMP_OFFLOAD_FLAGS} -c $<  -o $@

saxpy_gpu_loop: saxpy_gpu_loop.o
	$(CC)  $(LDFLAGS)  ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

saxpy_gpu_collapse.o: saxpy_gpu_collapse.c
	$(CC) $(CFLAGS) ${OPENMP_OFFLOAD_FLAGS} -c $<  -o $@

saxpy_gpu_collapse: saxpy_gpu_collapse.o
	$(CC)  $(LDFLAGS)  ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

clean:
	rm -f saxpy_gpu_singleunit_autoalloc saxpy_gpu_singleunit_dynamic
	rm -f saxpy_cpu saxpy_gpu_parallelfor saxpy_gpu_loop saxpy_gpu_collapse *.o
