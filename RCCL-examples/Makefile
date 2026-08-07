# Makefile for RCCL-examples

MPICXX    ?= mpicxx
ROCM_PATH ?= /opt/rocm

OMPI_CXX  ?= hipcc
export OMPI_CXX

CXXFLAGS = -O2 -I$(ROCM_PATH)/include
LDFLAGS  = -L$(ROCM_PATH)/lib -lrccl -Wl,-rpath,$(ROCM_PATH)/lib

EXECUTABLES = pt2pt collective

.PHONY: all test clean

all: $(EXECUTABLES)

pt2pt: pt2pt.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

collective: collective.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test: all
	mpirun -n 2 -mca pml ucx ./pt2pt
	mpirun -n 2 -mca pml ucx ./collective

clean:
	rm -f $(EXECUTABLES)
