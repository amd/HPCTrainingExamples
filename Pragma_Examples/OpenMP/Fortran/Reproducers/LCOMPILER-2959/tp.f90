! LCOMPILER-2959 reproducer (from NEC, unmodified).
! amdflang miscompiles at -O0 when a `parallel do` is nested inside a
! sequential do loop within a `target teams distribute` region: wrong results
! or a GPU memory access fault. -O1 and above are unaffected.
program tp
  implicit none
  integer, parameter :: ni=128, nj=128, nk=64
  integer :: i, j, k
  real(8), allocatable :: a(:,:,:), b(:,:,:)

  allocate(a(ni,nj,nk), b(ni,nj,nk))
  do k=1,nk; do j=1,nj; do i=1,ni
    a(i,j,k) = i + 1000*j + 1000000*k
  enddo; enddo; enddo
  b = 0d0

!$omp target teams distribute map(to:a) map(from:b)
  do k=1,nk
    do i=1,ni
!$omp parallel do
      do j=1,nj
        b(i,j,k) = a(i,j,k)
      enddo
!$omp end parallel do
    enddo
  enddo
!$omp end target teams distribute

  if (all(b == a)) then
    print *, 'PASS'
  else
    print *, 'FAIL: mismatches =', count(b /= a)
  endif
end program tp
