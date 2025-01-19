all: saxpy_gpu_target saxpy_gpu_target_teams saxpy_gpu_target_teams_distribute saxpy_gpu_target_parallel_for

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

OPENMP_FLAGS=-fopenmp
OPENMP_OFFLOAD_FLAGS=-fopenmp --offload-arch=${ROCM_GPU}

CFLAGS = -g -O3 -fstrict-aliasing

LDFLAGS = -lm

saxpy_gpu_target.o: saxpy_gpu_target.c
	$(CC) $(CFLAGS) ${OPENMP_OFFLOAD_FLAGS} -c $<  -o $@

saxpy_gpu_target: saxpy_gpu_target.o
	$(CC)  $(LDFLAGS)  ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

saxpy_gpu_target_teams.o: saxpy_gpu_target_teams.c
	$(CC) $(CFLAGS) ${OPENMP_OFFLOAD_FLAGS} -c $<  -o $@

saxpy_gpu_target_teams: saxpy_gpu_target_teams.o
	$(CC)  $(LDFLAGS)  ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

saxpy_gpu_target_teams_distribute.o: saxpy_gpu_target_teams_distribute.c
	$(CC) $(CFLAGS) ${OPENMP_OFFLOAD_FLAGS} -c $<  -o $@

saxpy_gpu_target_teams_distribute: saxpy_gpu_target_teams_distribute.o
	$(CC)  $(LDFLAGS)  ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

saxpy_gpu_target_parallel_for.o: saxpy_gpu_target_parallel_for.c
	$(CC) $(CFLAGS) ${OPENMP_OFFLOAD_FLAGS} -c $<  -o $@

saxpy_gpu_target_parallel_for: saxpy_gpu_target_parallel_for.o
	$(CC)  $(LDFLAGS)  ${OPENMP_OFFLOAD_FLAGS} $^ -o $@

clean:
	rm -f saxpy_gpu_target saxpy_gpu_target_teams
	rm -f saxpy_gpu_target_teams_distribute saxpy_gpu_parallelfor
	rm -f *.o
