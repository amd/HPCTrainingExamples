! This example was created by Johanna Potyka
! Copyright (c) 2024 AMD HPC Application Performance Team
! MIT License


       module computemod
          !--- device routine made public
          public :: compute

         contains
          !--- device routine
          subroutine compute(x)
              implicit none
              !$omp declare target
              !--------------------
              !example routine called from kernel
              !--- variables
              integer,parameter :: rk=8
              real(kind=rk),intent(inout) :: x
              !x               a value (from array)

              !---
              x = 1.0_rk

          end subroutine compute
      end module
