all: saxpy_gpu_target saxpy_gpu_target_teams saxpy_gpu_target_teams_distribute \
     saxpy_gpu_target_parallel_do saxpy_gpu_collapse saxpy_gpu_split_level \
     saxpy_gpu_parallel_do

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

FC1=$(notdir $(FC))

ifeq ($(findstring amdflang-new,$(FC1)), amdflang-new)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
  FREE_FORM_FLAG = -ffree-form
else ifeq ($(findstring flang,$(FC1)), flang)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
  FREE_FORM_FLAG = -Mfreeform
else ifeq ($(findstring gfortran,$(FC1)), gfortran)
  OPENMP_FLAGS = -fopenmp -foffload=-march=${ROCM_GPU} -fopt-info-optimized-omp
  FREE_FORM_FLAG = -ffree-form
else ifeq ($(findstring ftn,$(FC1)), ftn)
  OPENMP_FLAGS = -fopenmp
endif

ifeq ($(findstring ftn,$(FC1)), ftn)
  FFLAGS = -G2 -eZ -O3 ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
else ifeq ($(findstring amdflang-new,$(FC1)), amdflang-new)
  FFLAGS = -O3 -ffast-math ${FREE_FORM_FLAG} ${OPENMP_FLAGS}
else
  FFLAGS = -g -O3 ${FREE_FORM_FLAG} -fstrict-aliasing ${OPENMP_FLAGS}
endif
ifeq (${FC1},gfortran-13)
   LDFLAGS = -fno-lto
else ifeq ($(findstring ftn,$(FC1)), ftn)
   LDFLAGS = -L/opt/rh/gcc-toolset-12/root/usr/lib/gcc/x86_64-redhat-linux/12
endif

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
