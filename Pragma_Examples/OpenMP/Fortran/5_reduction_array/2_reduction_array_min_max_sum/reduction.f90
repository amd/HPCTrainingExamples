program rt003_reduction
  implicit none

  integer, parameter :: dp = kind(1.0d0)
  integer :: i, n, imax_result, imin_result, isum_result
  real(dp) :: dmax_result, dmin_result, dsum_result
  integer :: failures

  integer, allocatable :: iarr(:)
  real(dp), allocatable :: darr(:)

  failures = 0

  ! =============================================
  ! Test 1: integer MAX reduction -- small array
  ! =============================================
  n = 4
  allocate(iarr(n))
  iarr = (/ 258, 290, 320, 258 /)

  imax_result = -huge(imax_result)
  !$omp target
  !$omp teams loop collapse(1) default(shared) private(i) reduction(MAX: imax_result)
  do i = 1, n, 1
    imax_result = max(imax_result, iarr(i))
  end do
  !$omp end teams loop
  !$omp end target

  write(*,'(a,i0,a,i0)') "Test 1  int MAX (n=4):       result=", imax_result, "  expected=", 320
  if (imax_result /= 320) then
    write(*,'(a)') "  FAIL"
    failures = failures + 1
  else
    write(*,'(a)') "  PASS"
  end if
  deallocate(iarr)

  ! =============================================
  ! Test 2: integer MAX reduction -- larger array
  ! =============================================
  n = 1024
  allocate(iarr(n))
  do i = 1, n
    iarr(i) = mod(i * 7, 5000)
  end do
  iarr(512) = 99999

  imax_result = -huge(imax_result)
  !$omp target
  !$omp teams loop collapse(1) default(shared) private(i) reduction(MAX: imax_result)
  do i = 1, n, 1
    imax_result = max(imax_result, iarr(i))
  end do
  !$omp end teams loop
  !$omp end target

  write(*,'(a,i0,a,i0)') "Test 2  int MAX (n=1024):    result=", imax_result, "  expected=", 99999
  if (imax_result /= 99999) then
    write(*,'(a)') "  FAIL"
    failures = failures + 1
  else
    write(*,'(a)') "  PASS"
  end if
  deallocate(iarr)

  ! =============================================
  ! Test 3: integer MIN reduction
  ! =============================================
  n = 256
  allocate(iarr(n))
  do i = 1, n
    iarr(i) = 1000 + i
  end do
  iarr(100) = -42

  imin_result = huge(imin_result)
  !$omp target
  !$omp teams loop collapse(1) default(shared) private(i) reduction(MIN: imin_result)
  do i = 1, n, 1
    imin_result = min(imin_result, iarr(i))
  end do
  !$omp end teams loop
  !$omp end target

  write(*,'(a,i0,a,i0)') "Test 3  int MIN (n=256):     result=", imin_result, "  expected=", -42
  if (imin_result /= -42) then
    write(*,'(a)') "  FAIL"
    failures = failures + 1
  else
    write(*,'(a)') "  PASS"
  end if
  deallocate(iarr)

  ! =============================================
  ! Test 4: integer SUM reduction
  ! =============================================
  n = 100
  allocate(iarr(n))
  do i = 1, n
    iarr(i) = i
  end do

  isum_result = 0
  !$omp target
  !$omp teams loop collapse(1) default(shared) private(i) reduction(+: isum_result)
  do i = 1, n, 1
    isum_result = isum_result + iarr(i)
  end do
  !$omp end teams loop
  !$omp end target

  write(*,'(a,i0,a,i0)') "Test 4  int SUM (n=100):     result=", isum_result, "  expected=", 5050
  if (isum_result /= 5050) then
    write(*,'(a)') "  FAIL"
    failures = failures + 1
  else
    write(*,'(a)') "  PASS"
  end if
  deallocate(iarr)

  ! =============================================
  ! Test 5: real(dp) MAX reduction
  ! =============================================
  n = 512
  allocate(darr(n))
  do i = 1, n
    darr(i) = real(i, dp) * 0.1d0
  end do
  darr(256) = 9999.0d0

  dmax_result = -huge(dmax_result)
  !$omp target
  !$omp teams loop collapse(1) default(shared) private(i) reduction(MAX: dmax_result)
  do i = 1, n, 1
    dmax_result = max(dmax_result, darr(i))
  end do
  !$omp end teams loop
  !$omp end target

  write(*,'(a,f12.2,a,f12.2)') "Test 5  dp  MAX (n=512):     result=", dmax_result, "  expected=", 9999.0d0
  if (abs(dmax_result - 9999.0d0) > 1.0d-6) then
    write(*,'(a)') "  FAIL"
    failures = failures + 1
  else
    write(*,'(a)') "  PASS"
  end if
  deallocate(darr)

  ! =============================================
  ! Test 6: real(dp) MIN reduction
  ! =============================================
  n = 512
  allocate(darr(n))
  do i = 1, n
    darr(i) = real(i, dp) * 0.1d0
  end do
  darr(300) = -7777.0d0

  dmin_result = huge(dmin_result)
  !$omp target
  !$omp teams loop collapse(1) default(shared) private(i) reduction(MIN: dmin_result)
  do i = 1, n, 1
    dmin_result = min(dmin_result, darr(i))
  end do
  !$omp end teams loop
  !$omp end target

  write(*,'(a,f12.2,a,f12.2)') "Test 6  dp  MIN (n=512):     result=", dmin_result, "  expected=", -7777.0d0
  if (abs(dmin_result - (-7777.0d0)) > 1.0d-6) then
    write(*,'(a)') "  FAIL"
    failures = failures + 1
  else
    write(*,'(a)') "  PASS"
  end if
  deallocate(darr)

  ! =============================================
  ! Test 7: real(dp) SUM reduction
  ! =============================================
  n = 1000
  allocate(darr(n))
  do i = 1, n
    darr(i) = 1.0d0
  end do

  dsum_result = 0.0d0
  !$omp target
  !$omp teams loop collapse(1) default(shared) private(i) reduction(+: dsum_result)
  do i = 1, n, 1
    dsum_result = dsum_result + darr(i)
  end do
  !$omp end teams loop
  !$omp end target

  write(*,'(a,f12.2,a,f12.2)') "Test 7  dp  SUM (n=1000):    result=", dsum_result, "  expected=", 1000.0d0
  if (abs(dsum_result - 1000.0d0) > 1.0d-6) then
    write(*,'(a)') "  FAIL"
    failures = failures + 1
  else
    write(*,'(a)') "  PASS"
  end if
  deallocate(darr)

  ! =============================================
  ! Test 8: integer MAX with conditional (NEMO pattern)
  ! =============================================
  n = 4
  allocate(iarr(n))
  iarr = (/ 258, 290, 320, 258 /)

  imax_result = -huge(imax_result)
  !$omp target
  !$omp teams loop collapse(1) default(shared) private(i) reduction(MAX: imax_result)
  do i = 1, n, 1
    if (iarr(i) > 0) then
      imax_result = max(imax_result, iarr(i))
    end if
  end do
  !$omp end teams loop
  !$omp end target

  write(*,'(a,i0,a,i0)') "Test 8  int MAX cond (n=4):  result=", imax_result, "  expected=", 320
  if (imax_result /= 320) then
    write(*,'(a)') "  FAIL"
    failures = failures + 1
  else
    write(*,'(a)') "  PASS"
  end if
  deallocate(iarr)

  ! =============================================
  ! Summary
  ! =============================================
  write(*,'(a)') "---"
  if (failures == 0) then
    write(*,'(a)') "ALL TESTS PASSED"
  else
    write(*,'(i0,a)') failures, " TEST(S) FAILED"
  end if

end program rt003_reduction

