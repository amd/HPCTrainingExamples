! Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
! This software is distributed under the MIT License
!
! This example was created by Johanna Potyka
      !--- device routine
      subroutine compute(x)
          implicit none
          !$omp declare target link(compute) device_type(nohost)
          !--------------------
          !example routine called from kernel
          !--- variables
          integer,parameter :: rk=8
          real(kind=rk),intent(inout) :: x
          !x               a value (from array)

          !---
          x = 1.0_rk

      end subroutine compute
