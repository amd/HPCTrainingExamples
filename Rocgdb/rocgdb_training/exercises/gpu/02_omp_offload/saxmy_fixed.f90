! Copyright Advanced Micro Devices, Inc.
!
! SPDX-License-Identifier: MIT

! Reference solution for 02_omp_offload (compare with saxmy.f90): allocate,
! initialise and map x so the device has a valid array to read. With a=2, x=1, y=2
! the result is y=4.0, under any HSA_XNACK. See README.org.
! Build: amdflang -g -O0 -fopenmp --offload-arch=gfx942 -o saxmyf_fixed saxmy_fixed.f90
program saxmy_omp
  implicit none
  integer, parameter :: n = 256
  real :: a = 2.0
  real, allocatable :: x(:), y(:)
  integer :: i
  allocate(x(n), y(n))           ! x allocated and initialised ...
  x = 1.0
  y = 2.0
  !$omp target teams distribute parallel do map(to:x) map(tofrom:y)
  do i = 1, n                    ! ... and mapped to the device, so x(i) is valid
     y(i) = a*x(i)*y(i)
  end do
  !$omp end target teams distribute parallel do
  print *, 'y(1) =', y(1), '  y(n) =', y(n)
  deallocate(x, y)
end program saxmy_omp
