! Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
! SPDX-License-Identifier: MIT
!
! Demonstration of ROCTx markers in Fortran using hipfort bindings.
! This program shows how to use roctx range markers to annotate
! different phases of computation for profiling with rocprof.
!
program roctx_demo
  use hipfort_roctx
  use iso_c_binding
  implicit none

  integer, parameter :: N = 1000000
  real(8), allocatable :: A(:), B(:), C(:)
  real(8) :: sum_result
  integer :: i, ret

  print *, "ROCTx Markers Demo (hipfort version)"
  print *, "====================================="
  print *, ""

  ret = roctxRangePush("initialization"//c_null_char)

  allocate(A(N), B(N), C(N))
  do i = 1, N
    A(i) = real(i, 8)
    B(i) = real(N - i + 1, 8)
  end do
  C = 0.0d0

  ret = roctxRangePop()

  print *, "Initialized arrays of size:", N

  ret = roctxRangePush("computation"//c_null_char)
  call roctxMark("starting vector add"//c_null_char)

  do i = 1, N
    C(i) = A(i) + B(i)
  end do

  call roctxMark("vector add complete"//c_null_char)

  print *, "Computed vector addition"

  ret = roctxRangePush("reduction"//c_null_char)

  sum_result = 0.0d0
  do i = 1, N
    sum_result = sum_result + C(i)
  end do

  ret = roctxRangePop()
  ret = roctxRangePop()

  print *, "Sum of C:", sum_result
  print *, "Expected:", real(N, 8) * real(N + 1, 8)
  print *, ""

  if (abs(sum_result - real(N, 8) * real(N + 1, 8)) < 1.0d-6) then
    print *, "SUCCESS"
  else
    print *, "FAILED"
    stop 1
  end if

  deallocate(A, B, C)

end program roctx_demo
