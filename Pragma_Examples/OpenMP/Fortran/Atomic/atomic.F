! This program is based on a reproducer created by Mahdieh Ghazimirsaeed
! Copyright (c) 2025 AMD HPC Application Performance Team
! MIT License

program atomic_vs_reduction
  use omp_lib
  implicit none

  integer, parameter :: real8 = kind(0.0d0)
  integer, parameter :: N = 10000
  real(real8) :: a(N)
  real(real8) :: sum_of_a, tstart
  integer :: i

  !$omp requires unified_shared_memory

  !$omp target teams distribute parallel do
  do i = 1, N
     a(i) = 1.0_real8
  end do
  !$omp end target teams distribute parallel do

  sum_of_a = 0.0_real8
  tstart = omp_get_wtime()
  !$omp target teams distribute parallel do map(tofrom:sum_of_a)
  do i = 1, N
     !$omp atomic
     sum_of_a = sum_of_a + a(i)
  end do
  !$omp end target teams distribute parallel do
  print *, '   Atomic result: ', sum_of_a, ' Runtime is: ', omp_get_wtime() - tstart, ' secs'

  sum_of_a = 0.0_real8
  tstart = omp_get_wtime()
  !$omp target teams distribute parallel do reduction(+:sum_of_a)
  do i = 1, N
     sum_of_a = sum_of_a + a(i)
  end do
  !$omp end target teams distribute parallel do
  print *, 'Reduction result: ', sum_of_a, ' Runtime is: ', omp_get_wtime() - tstart, ' secs'

end program atomic_vs_reduction
