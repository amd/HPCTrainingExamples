ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//')) 
HSA_XNACK ?= 1
OMP_TARGET_OFFLOAD ?= MANDATORY

SRC := reduction.f90
EXE := reduction

FC1=$(notdir $(FC))

ifneq ($(findstring amdflang,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
else ifneq ($(findstring flang,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
else ifneq ($(findstring gfortran,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload=-march=$(ROCM_GPU)
  #OPENMP_FLAGS = -fopenmp -foffload=-march=${ROCM_GPU} -fopt-info-optimized-omp
else ifneq ($(findstring ftn,$(FC1)),)
  FTN_VERSION := $(shell $(FC) --version 2>&1)
  OPENMP_FLAGS = -fopenmp
  ifneq ($(findstring Cray,$(FTN_VERSION)),)
    FFLAGS = -eZ
  else
    FFLAGS =
  endif
endif

FFLAGS := -g -O3 -fdefault-real-8 -fdefault-double-8 ${FFLAGS} ${OPENMP_FLAGS}
LDFLAGS = ${OPENMP_FLAGS}

.PHONY: all clean run check

all: $(EXE)

$(EXE): $(SRC)
	$(FC) $(FFLAGS) -o $@ $< $(LINK_FLAGS) $(LDFLAGS) $(LDLIBS)

run: $(EXE)
	HSA_XNACK=$(HSA_XNACK) OMP_TARGET_OFFLOAD=$(OMP_TARGET_OFFLOAD) ./$(EXE)

check: run

clean:
	rm -f $(EXE)

