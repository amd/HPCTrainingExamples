##**************************************************************************
##* Copyright (c) 2022, Advanced Micro Devices, Inc. All rights reserved.
##**************************************************************************

ROCM_PATH?=/opt/rocm

# Compilers
MPICC=$(PREP) mpic++
MPILD=$(PREP) $(ROCM_PATH)/bin/hipcc
HIPCC=$(PREP) $(ROCM_PATH)/bin/hipcc

IS_OMPI=$(shell mpirun --version | grep "Open MPI")
IS_MPICH=$(shell mpirun --version | grep mpich)

ifneq ($(IS_OMPI),)
  MPICFLAGS=$(shell mpic++ --showme:compile)
  MPILDFLAGS=$(shell mpic++ --showme:link)
else ifneq ($(IS_MPICH),)
  MPICFLAGS=$(shell mpic++ -compile_info | cut -d" " -f 2-)
  MPILDFLAGS=$(shell mpic++ -link_info | cut -d" " -f 2-)
else
  $(error Unknown MPI version! Currently can detect mpich or open-mpi)
endif

# Flags
CFLAGS=-O3 -g -ggdb -fPIC -std=c++11 -march=native -Wall
CFLAGS+=-I$(ROCM_PATH)roctracer/include
LDFLAGS=-L$(ROCM_PATH)lib/ -lroctx64 -lroctracer64
MPICFLAGS+=$(CFLAGS)
MPILDFLAGS+=$(LDFLAGS)
HIPFLAGS=-lamdhip64
HIPCFLAGS=$(shell $(ROCM_PATH)/bin/hipconfig --cpp_config)

# Description of binaries
BINDIR=.
JACOBI_HIP=$(BINDIR)/Jacobi_hip
BINARIES=$(JACOBI_HIP)

DEPS=Jacobi.hpp \
		 defines.hpp \
		 markers.h

OBJS=JacobiSetup.o  \
		 JacobiRun.o    \
		 JacobiMain.o   \
		 HaloExchange.o \
		 Input.o        \
		 JacobiIteration.o \
		 Laplacian.o    \
		 Norm.o

# Commands
all: $(BINARIES)

HaloExchange.o: $(DEPS)  HaloExchange.hip
	$(HIPCC) $(MPICFLAGS) $(HIPCFLAGS) $(CFLAGS) -c HaloExchange.hip -o HaloExchange.o

Input.o: $(DEPS)  Input.hip
	$(HIPCC) $(MPICFLAGS) $(HIPCFLAGS) $(CFLAGS) -c Input.hip -o Input.o

JacobiMain.o: $(DEPS)  JacobiMain.hip
	$(HIPCC) $(MPICFLAGS) $(HIPCFLAGS) $(CFLAGS) -c JacobiMain.hip -o JacobiMain.o

JacobiRun.o: $(DEPS)  JacobiRun.hip
	$(HIPCC) $(MPICFLAGS) $(HIPCFLAGS) $(CFLAGS) -c JacobiRun.hip -o JacobiRun.o

JacobiSetup.o: $(DEPS)  JacobiSetup.hip
	$(HIPCC) $(MPICFLAGS) $(HIPCFLAGS) $(CFLAGS) -c JacobiSetup.hip -o JacobiSetup.o

JacobiIteration.o: $(DEPS)  JacobiIteration.hip
	$(HIPCC) $(MPICFLAGS) $(CFLAGS) -c JacobiIteration.hip -o JacobiIteration.o

Laplacian.o: $(DEPS)  Laplacian.hip
	$(HIPCC) $(MPICFLAGS) $(CFLAGS) -c Laplacian.hip -o Laplacian.o

Norm.o: $(DEPS)  Norm.hip
	$(HIPCC) $(MPICFLAGS) $(CFLAGS) -c Norm.hip -o Norm.o

$(JACOBI_HIP): $(OBJS)
	$(MPILD) $(MPICFLAGS) $(CFLAGS) -o $(JACOBI_HIP)  $(OBJS) $(MPILDFLAGS) $(LDFLAGS)

clean:
	rm -rf *.o *~ $(BINARIES)
