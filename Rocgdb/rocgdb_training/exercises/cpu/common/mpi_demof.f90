! Copyright Advanced Micro Devices, Inc.
!
! SPDX-License-Identifier: MIT

program mpi_demo
  use mpi
  implicit none
  integer :: rank, size, value, ierr

  call MPI_Init(ierr)

  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)

  value = rank * rank   ! a per-rank value to break on and inspect
  print '(a, i0, a, i0, a, i0)', "rank ", rank, " of ", size, ": value = ", value

  call MPI_Finalize(ierr)
end program mpi_demo
