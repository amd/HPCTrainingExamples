! Program to act as MWE for potential compile bug with amdflang + fortran
program mwe
  use omp_lib
  implicit none
  !$OMP requires unified_shared_memory
  integer :: i,j,nmax
  real, allocatable, dimension(:) :: rhoi

  ! initialise for loop
  nmax = 100
  allocate(rhoi(nmax))
  rhoi(:) = 0.

!$OMP target teams distribute default(none) &
!$OMP private(i,j) &
!$OMP shared(rhoi,nmax)

  do j = 1,100
    ! Lines 24-26 works when the -g flag is included in compilation,
    ! but Lines 29 does not.

    ! e.g. the below works with -g ...
#ifdef ITWORKS
    do i = 1,nmax
      rhoi(i) = 1.0
    enddo
#else
    ! but using the line below instead does not
    rhoi(1:nmax) = 1.0
#endif

  enddo

!$OMP end target teams distribute

  print*,rhoi(1:10)

  deallocate(rhoi)

end program mwe
