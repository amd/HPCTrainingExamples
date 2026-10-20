! Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
! This software is distributed under the MIT License
!
! Demonstrate OpenMP 6.0 workdistribute with whole-array assignments.
program workdistribute_arrays
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none

   integer, parameter :: nx = 1024
   integer, parameter :: ny = 512
   real(real64):: src(nx,ny), dst(nx,ny)
   integer :: i, j


   !nonsense value initialized on host to check
   dst = -9999999_real64

   !array initialization on GPU
   !$omp target teams workdistribute
   src = 42.0_real64
   !$omp end target teams workdistribute

   !array copy on GPU
   !$omp target teams workdistribute
   dst = src
   !$omp end target teams workdistribute

   write(*,*) "dst(1,1)",dst(1,1)
   write(*,*) "dst(nx,ny)",dst(nx,ny)
   if(ALL(ABS(dst-42.0_real64) < 1e-15_real64)) write(*,*) "result correct"
end program workdistribute_arrays
