! Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
! This software is distributed under the MIT License
!
! Starting point: sum the elements of an array (each set to 1.0) using a
! serial do loop. Port this to "do concurrent" and use REDUCE(+) for the
! summation, then build with the device do-concurrent and OpenMP offload
! flags in `1_do_concurrent_reduce`.
!
program freduce
   implicit none
   integer, parameter :: n = 100000
   integer :: j
   real(kind=8) :: sum2
   real(kind=8), allocatable :: array(:)

   allocate(array(n))
   do j = 1, n
      array(j) = 1.0_8
   end do

   sum2 = 0.0_8
   do j = 1, n
      sum2 = sum2 + array(j)
   end do

   write (*, *) "sum=", sum2
   deallocate(array)
end program freduce
