##**************************************************************************
##* Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
##**************************************************************************

# Compilers and related
ROCM_PATH     ?= /opt/rocm
ROCM_GPU      ?= $(INSTALLED_GPU)
INSTALLED_GPU = $(shell $(ROCM_PATH)/bin/rocm_agent_enumerator | grep -m 1 -E gfx[^0]{1})
CFLAG         = -Ofast -g --offload-arch=$(ROCM_GPU)
LDFLAG        =

# OpenMP
CFLAGS    = -fopenmp
LDFLAGS   =



# Source code
OBJS= Jacobi.o \
  Laplacian.o \
	BoundaryConditions.o \
  Update.o \
	Norm.o \
	Input.o \
	Main.o

# Commands
all: jacobi


jacobi: $(OBJS)
	$(CC) $(CFLAG) $(CFLAGS) $(LDFLAG) $(LDFLAGS) -o Jacobi  $(OBJS)

%.o : %.cpp
	$(CC) $(CFLAG) $(CFLAGS) -c $< -o $@


clean: 
	rm -rf Jacobi *.o
