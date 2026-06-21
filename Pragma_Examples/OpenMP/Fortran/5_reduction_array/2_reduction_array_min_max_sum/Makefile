GPU_ARCH ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//')) 
HSA_XNACK ?= 1
OMP_TARGET_OFFLOAD ?= MANDATORY

SRC := reduction.f90
EXE := reduction

FC1=$(notdir $(FC))

ifneq ($(findstring amdflang,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(GPU_ARCH)
else ifneq ($(findstring flang,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(GPU_ARCH)
else ifneq ($(findstring gfortran,$(FC1)),)
  OPENMP_FLAGS = -fopenmp --offload=-march=$(GPU_ARCH)
else ifneq ($(findstring ftn,$(FC1)),)
  FTN_VERSION := $(shell $(FC) --version 2>&1)
  OPENMP_FLAGS = -fopenmp
endif

FFLAGS := -g -O3 -fdefault-real-8 -fdefault-double-8 ${OPENMP_FLAGS}
LDFLAGS := ${OPENMP_FLAGS}

.PHONY: all clean run check

all: $(EXE)

$(EXE): $(SRC)
	$(FC) $(FFLAGS) -o $@ $< $(LINK_FLAGS) $(LDFLAGS) $(LDLIBS)

run: $(EXE)
	HSA_XNACK=$(HSA_XNACK) OMP_TARGET_OFFLOAD=$(OMP_TARGET_OFFLOAD) ./$(EXE)

check: run

clean:
	rm -f $(EXE)

