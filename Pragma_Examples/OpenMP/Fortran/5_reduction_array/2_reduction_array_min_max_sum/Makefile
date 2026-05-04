FC := amdflang
GPU_ARCH ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//')) 
HSA_XNACK ?= 1
OMP_TARGET_OFFLOAD ?= MANDATORY

SRC := reduction.f90
EXE := reduction

COMMON_FLAGS := -fdefault-real-8 -fdefault-double-8 -fopenmp --offload-arch=$(GPU_ARCH) -O3 -g
LINK_FLAGS := -fopenmp --offload-arch=$(GPU_ARCH)

.PHONY: all clean run check

all: $(EXE)

$(EXE): $(SRC)
	$(FC) $(COMMON_FLAGS) $(FFLAGS) -o $@ $< $(LINK_FLAGS) $(LDFLAGS) $(LDLIBS)

run: $(EXE)
	HSA_XNACK=$(HSA_XNACK) OMP_TARGET_OFFLOAD=$(OMP_TARGET_OFFLOAD) ./$(EXE)

check: run

clean:
	rm -f $(EXE)

