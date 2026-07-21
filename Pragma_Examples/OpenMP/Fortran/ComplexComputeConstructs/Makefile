all: saxpy_gpu_target saxpy_gpu_target_teams saxpy_gpu_target_teams_distribute \
     saxpy_gpu_target_parallel_do saxpy_gpu_collapse saxpy_gpu_split_level \
     saxpy_gpu_parallel_do

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

FC1=$(notdir $(FC))

ifneq ($(findstring amdflang,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
  FREE_FORM_FLAG = -ffree-form
else ifneq ($(findstring flang,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
  FREE_FORM_FLAG = -ffree-form
else ifneq ($(findstring gfortran,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload=-march=$(ROCM_GPU)
  #OPENMP_FLAGS = -fopenmp -foffload=-march=${ROCM_GPU} -fopt-info-optimized-omp
  FREE_FORM_FLAG = -ffree-form
else ifneq ($(findstring ftn,$(FC1)),)
  FTN_VERSION := $(shell $(FC) --version 2>&1)
  OPENMP_FLAGS = -fopenmp
  ifneq ($(findstring Cray,$(FTN_VERSION)),)
    FREE_FORM_FLAG = -f free
    FFLAGS = -eZ
  else
    FREE_FORM_FLAG = -ffree-form
    FFLAGS =
  endif
endif

FFLAGS := -g -O3 ${FFLAGS} ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
LDFLAGS = ${OPENMP_FLAGS}

saxpy_gpu_target.o: saxpy_gpu_target.F90
	$(FC) $(FFLAGS) ${OPENMP_FLAGS} -c $<  -o $@

saxpy_gpu_target: saxpy_gpu_target.o
	$(FC)  $(LDFLAGS)  ${OPENMP_FLAGS} $^ -o $@

saxpy_gpu_target_teams.o: saxpy_gpu_target_teams.F90
	$(FC) $(FFLAGS) ${OPENMP_FLAGS} -c $<  -o $@

saxpy_gpu_target_teams: saxpy_gpu_target_teams.o
	$(FC)  $(LDFLAGS)  ${OPENMP_FLAGS} $^ -o $@

saxpy_gpu_target_teams_distribute.o: saxpy_gpu_target_teams_distribute.F90
	$(FC) $(FFLAGS) ${OPENMP_FLAGS} -c $<  -o $@

saxpy_gpu_target_teams_distribute: saxpy_gpu_target_teams_distribute.o
	$(FC)  $(LDFLAGS)  ${OPENMP_FLAGS} $^ -o $@

saxpy_gpu_target_parallel_do.o: saxpy_gpu_target_parallel_do.F90
	$(FC) $(FFLAGS) ${OPENMP_FLAGS} -c $<  -o $@

saxpy_gpu_target_parallel_do: saxpy_gpu_target_parallel_do.o
	$(FC)  $(LDFLAGS)  ${OPENMP_FLAGS} $^ -o $@

saxpy_gpu_parallel_do.o: saxpy_gpu_parallel_do.F90
	$(FC) $(FFLAGS) ${OPENMP_FLAGS} -c $<  -o $@

saxpy_gpu_parallel_do: saxpy_gpu_parallel_do.o
	$(FC)  $(LDFLAGS)  ${OPENMP_FLAGS} $^ -o $@

saxpy_gpu_collapse.o: saxpy_gpu_collapse.F90
	$(FC) $(FFLAGS) ${OPENMP_FLAGS} -c $<  -o $@

saxpy_gpu_collapse: saxpy_gpu_collapse.o
	$(FC)  $(LDFLAGS)  ${OPENMP_FLAGS} $^ -o $@

saxpy_gpu_split_level.o: saxpy_gpu_split_level.F90
	$(FC) $(FFLAGS) ${OPENMP_FLAGS} -c $<  -o $@

saxpy_gpu_split_level: saxpy_gpu_split_level.o
	$(FC)  $(LDFLAGS)  ${OPENMP_FLAGS} $^ -o $@

clean:
	rm -f saxpy_gpu_target saxpy_gpu_target_teams saxpy_gpu_target_parallel_do
	rm -f saxpy_gpu_target_teams_distribute saxpy_gpu_parallel_do
	rm -f saxpy_gpu_collapse saxpy_gpu_split saxpy_gpu_split_level
	rm -rf *.o *.mod
