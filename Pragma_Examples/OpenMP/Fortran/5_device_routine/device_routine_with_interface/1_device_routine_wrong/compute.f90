! This example was created by Johanna Potyka
! Copyright (c) 2025 AMD HPC Application Performance Team
! MIT License

      !--- device routine
      subroutine compute(x)
          implicit none
          !$omp requires unified_shared_memory
          !--------------------
          !example routine called from kernel
          !--- variables
          integer,parameter :: rk=8
          real(kind=rk),intent(inout) :: x
          !x               a value (from array)

          !---
          x = 1.0_rk

      end subroutine compute
