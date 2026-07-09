##**************************************************************************
##* Copyright (c) 2022, Advanced Micro Devices, Inc. All rights reserved.
##**************************************************************************

# Description of binaries
BINDIR=.
JACOBI_HIP=$(BINDIR)/Jacobi_hip
BINARIES=$(JACOBI_HIP)

.PHONY: all clean
default: $(BINARIES)
all: $(BINARIES)
clean:
	rm -rf *.o *~ $(BINARIES)


ROCM_PATH?=/opt/rocm

# Compilers
MPICC=$(PREP) mpic++
MPILD=$(PREP) $(ROCM_PATH)/bin/hipcc
HIPCC=$(PREP) $(ROCM_PATH)/bin/hipcc

USE_CRAYPE=$(shell mpic++ --version | grep "CrayPE is loaded")

ifneq ($(USE_CRAYPE),)
  MPICC=CC
endif

IS_OMPI=$(shell which ompi_info 2>/dev/null)
# MPICH detection. The Cray cc/CC wrapper prints "CrayPE is loaded" (matched by
# 'Cray' below). The standalone mpich-wrappers MPICH on a Cray, however, reports
# the underlying compiler from `--version` (e.g. "AMD clang"), so grepping the
# version string misses it -- fall back to a working MPICH-style `-compile_info`
# (mpich-wrappers supports it; see the else branch), then to the `mpichversion`
# tool. (Previously the second assignment clobbered the first, so only the
# 'Cray' match survived and mpich-wrappers hit the "Unknown MPI version" error.)
IS_MPICH=$(shell ${MPICC} --version 2>/dev/null | grep -iE 'mpich|cray')
ifeq ($(strip $(IS_MPICH)),)
  IS_MPICH=$(shell ${MPICC} -compile_info >/dev/null 2>&1 && echo mpich)
endif
ifeq ($(strip $(IS_MPICH)),)
  IS_MPICH=$(shell which mpichversion 2>/dev/null)
endif

ifneq ($(IS_OMPI),)
  MPICFLAGS=$(shell ${MPICC} --showme:compile)
  MPILDFLAGS=$(shell ${MPICC} --showme:link)
else ifneq ($(IS_MPICH),)
    ifneq ($(USE_CRAYPE),)
       MPICFLAGS=$(shell ${MPICC} --cray-print-opts=cflags)
       MPILDFLAGS=$(shell ${MPICC} --cray-print-opts=libs)
    else
       MPICFLAGS=$(shell ${MPICC} -compile_info | cut -d" " -f 2-)
       MPILDFLAGS=$(shell ${MPICC} -link_info | cut -d" " -f 2-)
    endif
else
    $(error Unknown MPI version! Currently can detect mpich or openmpi)
endif

# Flags
CFLAGS=-O3 -g -ggdb -fPIC -std=c++11 -march=native -Wall
CFLAGS+=-I$(ROCM_PATH)/roctracer/include -I${ROCM_PATH}/include/roctracer
LDFLAGS=-L$(ROCM_PATH)/lib/ -lroctx64 -lroctracer64
MPICFLAGS+=$(CFLAGS)
MPILDFLAGS+=$(LDFLAGS)
HIPFLAGS=-lamdhip64
HIPCFLAGS=$(shell $(ROCM_PATH)/bin/hipconfig --cpp_config)

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
