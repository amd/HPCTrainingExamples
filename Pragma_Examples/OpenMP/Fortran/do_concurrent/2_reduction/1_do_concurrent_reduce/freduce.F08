! Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
! This software is distributed under the MIT License
!
program freduce
   implicit none
   integer, parameter :: n = 100000
   integer :: j
   real(kind=8) :: sum2
   real(kind=8), allocatable :: array(:)

   allocate(array(n))

   do concurrent (j = 1:n)
      array(j) = 1.0_8
   end do

   sum2 = 0.0_8
   do concurrent (j = 1:n) REDUCE (+: sum2)
      sum2 = sum2 + array(j)
   end do

   write (*, *) "sum=", sum2
   deallocate(array)
end program freduce
