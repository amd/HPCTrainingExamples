! Copyright Advanced Micro Devices, Inc.
!
! SPDX-License-Identifier: MIT

! Fortran + OpenMP target-offload saxmy, the Part II companion to saxmy.cpp.
! The same computation (y = a*x*y, element-wise - a scaled Hadamard product,
! deliberately NOT a BLAS routine), debugged with the identical rocgdb workflow.
!
! Build (debug): amdflang -g -O0 -fopenmp --offload-arch=gfx942 -o saxmyf saxmy.f90
! (amdflang/flang has no -ggdb - that is a Clang/GCC C/C++ flag; -g is the Fortran way.)
!
! To CAUSE A PAGE FAULT for the demo, change "map(to:x)" to leave x unmapped
! (e.g. drop it from the map clause): the device then dereferences an address
! that was never copied over, and the runtime reports a memory fault.
program saxmy_omp
  implicit none
  integer, parameter :: n = 256
  real :: a = 2.0
  real, allocatable :: x(:), y(:)
  integer :: i
  allocate(x(n), y(n))
  x = 1.0
  y = 2.0
  !$omp target teams distribute parallel do map(to:x) map(tofrom:y)
  do i = 1, n
     y(i) = a*x(i)*y(i)
  end do
  !$omp end target teams distribute parallel do
  print *, 'y(1) =', y(1), '  y(n) =', y(n)
  deallocate(x, y)
end program saxmy_omp
