! LCOMPILER-1858 reproducer (from the EPCC hackathon, via Giacomo Capodaglio).
! Passing arrays as private to a target region errors, but only at -O0
! (adding thread_limit(256) hides it for small arrays). The failure is at
! compile time with the 6.1.0 amdflang-new drop; this test therefore checks
! that the program builds and runs.
program private_arrays
  implicit none
  integer :: i
  real*8 :: dra(1:20)
  real*8 :: dlf(1:20), ddlf(1:20)
  !$omp target teams distribute parallel do default(shared) &
  !$omp   private(dra, dlf, ddlf) dist_schedule(static)
  do i = 1, 20
     dra  = 2.0
     dlf  = 2.0
     ddlf = 2.0
  end do
  !$omp end target teams distribute parallel do
  print *, dra(20), dlf(20), ddlf(20)
end program private_arrays
