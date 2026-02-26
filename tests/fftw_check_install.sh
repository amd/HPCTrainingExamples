#!/bin/bash

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
   echo "rocm module is not loaded"
   echo "loading default rocm module"
   module load rocm
fi

module load openmpi
module load fftw

FFTW_INCLUDE=${FFTW_PATH}/include

echo "=== FFTW MPI Fortran interface test ==="
echo "FFTW_PATH: ${FFTW_PATH}"
echo "Fortran compiler wrapped by mpifort:"
mpifort -show

cat > test_fftw_mpi.f90 << 'EOF'
program test_fftw_mpi
  use, intrinsic :: iso_c_binding
  use MPI
  implicit none
  include 'fftw3-mpi.f03'

  integer :: ierr, rank
  integer(C_INTPTR_T) :: N, local_n0, local_0_start, alloc_local

  N = 64

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)

  call fftw_mpi_init()

  alloc_local = fftw_mpi_local_size_2d(N, N, MPI_COMM_WORLD, &
       local_n0, local_0_start)

  if (rank == 0) then
     print *, 'FFTW Install Check: SUCCESS'
     print *, 'alloc_local:', alloc_local
     print *, 'local_n0:', local_n0, 'local_0_start:', local_0_start
  end if

  call fftw_mpi_cleanup()
  call MPI_Finalize(ierr)

end program test_fftw_mpi
EOF

echo "=== Compiling test_fftw_mpi.f90 ==="
mpifort test_fftw_mpi.f90 -o test_fftw_mpi \
  -I${FFTW_INCLUDE} \
  -L${FFTW_PATH}/lib -lfftw3_mpi -lfftw3 -lm

echo "=== Running test ==="
mpirun -n 2 ./test_fftw_mpi

echo "=== Cleaning up ==="
rm -f test_fftw_mpi.f90 test_fftw_mpi
