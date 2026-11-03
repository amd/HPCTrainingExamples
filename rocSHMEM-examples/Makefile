# Makefile for rocSHMEM-examples

HIPCC            ?= hipcc
ROCM_PATH        ?= /opt/rocm
ROCM_GPU         ?= $(strip $(shell rocminfo | grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

MPI_CFLAGS  := $(shell mpicxx --showme:compile)
MPI_LDFLAGS := $(shell mpicxx --showme:link)

ifndef ROCSHMEM_PATH
$(error ROCSHMEM_PATH is not set. Run: module load rocshmem/3.2.0-ro_ipc)
endif

HIPFLAGS = --offload-arch=$(ROCM_GPU)
CXXFLAGS = -O2 -I$(ROCM_PATH)/include -I$(ROCSHMEM_PATH)/include $(MPI_CFLAGS)
LDFLAGS  = $(ROCSHMEM_PATH)/lib/librocshmem.a $(MPI_LDFLAGS) -L$(ROCM_PATH)/lib -lamdhip64 -lhsa-runtime64

EXECUTABLES = onesided collective

.PHONY: all test clean

all: $(EXECUTABLES)

%.o: %.cpp
	$(HIPCC) -c -fgpu-rdc -x hip $(HIPFLAGS) $(CXXFLAGS) -o $@ $<

onesided: onesided.o
	$(HIPCC) -fgpu-rdc --hip-link $(HIPFLAGS) -o $@ $< $(LDFLAGS)

collective: collective.o
	$(HIPCC) -fgpu-rdc --hip-link $(HIPFLAGS) -o $@ $< $(LDFLAGS)

test: all
	mpirun -np 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=2 ./onesided
	mpirun -np 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=2 ./collective

clean:
	rm -f $(EXECUTABLES) *.o
