# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

# Shared programs for the hands-on exercises.
# Build the (non-MPI) programs with:   make          (C via amdclang, Fortran via amdflang)
# Build the MPI programs separately with: make mpi    (needs an MPI compiler)

CC     = amdclang
CFLAGS = -ggdb -O0
FC     = amdflang
# amdflang (flang) has no -ggdb (that is a Clang/GCC C/C++ flag); plain -g only.
FFLAGS = -g -O0
PROGS  = demo floats faulty
FPROGS = demof floatsf faultyf

# `fusion`/`fusionf` are built WITH optimization on purpose (that is the whole
# point of the wrong-line exercise), so they are listed separately from PROGS.
.DEFAULT_GOAL := all
all: $(PROGS) fusion $(FPROGS) fusionf

$(PROGS): %: %.c
	$(CC) $(CFLAGS) -o $@ $<

$(FPROGS): %: %.f90
	$(FC) $(FFLAGS) -o $@ $<

fusion: fusion.c
	$(CC) -ggdb -O2 -o $@ $<

fusionf: fusionf.f90
	$(FC) -g -O2 -o $@ $<

# MPI programs - separate targets so a plain `make` still works on a machine
# without an MPI stack (e.g. a login node).
MPICC = mpicc
MPIFC = mpif90
mpi: mpi_demo.c mpi_demof.f90
	$(MPICC) -ggdb -O0 -o mpi_demo mpi_demo.c
	$(MPIFC) -g -O0 -o mpi_demof mpi_demof.f90

clean:
	rm -f $(PROGS) fusion $(FPROGS) fusionf mpi_demo mpi_demof *.mod

.PHONY: all mpi clean
