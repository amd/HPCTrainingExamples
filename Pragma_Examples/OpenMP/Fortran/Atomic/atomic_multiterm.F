! Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
! This software is distributed under the MIT License
!
! Multi-term atomic update: when the right-hand side adds more than one term
! to the atomic variable, the atomic variable must be a direct argument of the
! top-level operator. Parenthesize the extra terms so the update parses as
! x = x + (b + c) rather than x = (x + b) + c.
program atomic_multiterm
  use omp_lib
  implicit none

  !$omp requires unified_shared_memory

  integer, parameter :: real8 = kind(0.0d0)
  integer, parameter :: N = 10000
  real(real8) :: a(N), b(N)
  real(real8) :: sum_of_a
  integer :: i

  !$omp target teams distribute parallel do
  do i = 1, N
     a(i) = 1.0_real8
     b(i) = 2.0_real8
  end do
  !$omp end target teams distribute parallel do

  sum_of_a = 0.0_real8
  !$omp target teams distribute parallel do map(tofrom:sum_of_a)
  do i = 1, N
     !$omp atomic
     sum_of_a = sum_of_a + (a(i) + b(i))
     ! Incorrect: sum_of_a = sum_of_a + a(i) + b(i)
     ! parses as (sum_of_a + a(i)) + b(i), so the atomic variable is not a
     ! direct argument of the top-level +; amdflang rejects it as a semantic error.
  end do
  !$omp end target teams distribute parallel do
  print *, 'Atomic result: ', sum_of_a, ' Expected: ', 3.0_real8 * N

  if (abs(sum_of_a - 3.0_real8 * N) < 1.e-10) then
     print*, 'PASS!'
  else
     print*, 'FAIL!'
  end if

end program atomic_multiterm
