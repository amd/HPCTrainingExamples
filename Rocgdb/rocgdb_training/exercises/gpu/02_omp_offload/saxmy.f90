! Copyright Advanced Micro Devices, Inc.
!
! SPDX-License-Identifier: MIT

! Fortran + OpenMP target-offload saxmy: y(i) = a*x(i)*y(i), the Part II companion
! to the HIP saxmy.cpp (a scaled Hadamard product, deliberately NOT a BLAS routine).
!
! THE BUG (the exercise): x is never allocated, so it is a null pointer, and the
! !$omp target region reads x(i) - the device dereferences address 0 and faults.
! An *allocated* x would be silently demand-paged onto the MI300A even while
! unmapped, hiding the fault; a null x faults reliably, under any HSA_XNACK. The
! reference solution is saxmy_fixed.f90 (allocate, initialise and map x). See
! README.org. Build: amdflang -g -O0 -fopenmp --offload-arch=gfx942 -o saxmyf saxmy.f90
program saxmy_omp
  implicit none
  integer, parameter :: n = 256
  real :: a = 2.0
  real, allocatable :: x(:), y(:)
  integer :: i
  allocate(y(n))                 ! BUG: x is never allocated -> null pointer
  y = 2.0
  !$omp target teams distribute parallel do map(tofrom:y)
  do i = 1, n
     y(i) = a*x(i)*y(i)          ! device reads x(i) from a null pointer -> fault
  end do
  !$omp end target teams distribute parallel do
  print *, 'y(1) =', y(1), '  y(n) =', y(n)
  deallocate(y)
end program saxmy_omp
