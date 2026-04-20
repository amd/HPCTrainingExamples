! Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
! SPDX-License-Identifier: MIT
!
! Demonstration of ROCTx markers in Fortran using standalone C bindings.
! This program shows how to use roctx range markers to annotate
! different phases of computation for profiling with rocprof.
!
! The roctx_mod module below provides the C interface to the roctx library.
! This is equivalent to what hipfort_roctx provides, but standalone.
!

module roctx_mod
  use iso_c_binding
  implicit none

  private
  public :: roctx_range_push, roctx_range_pop, roctx_mark

  interface
    function roctxRangePushA(message) bind(C, name="roctxRangePushA") result(ret)
      import :: c_int, c_char
      character(kind=c_char), intent(in) :: message(*)
      integer(c_int) :: ret
    end function roctxRangePushA

    function roctxRangePop() bind(C, name="roctxRangePop") result(ret)
      import :: c_int
      integer(c_int) :: ret
    end function roctxRangePop

    subroutine roctxMarkA(message) bind(C, name="roctxMarkA")
      import :: c_char
      character(kind=c_char), intent(in) :: message(*)
    end subroutine roctxMarkA
  end interface

contains

  function roctx_range_push(message) result(ret)
    character(len=*), intent(in) :: message
    integer :: ret
    ret = roctxRangePushA(trim(message)//c_null_char)
  end function roctx_range_push

  function roctx_range_pop() result(ret)
    integer :: ret
    ret = roctxRangePop()
  end function roctx_range_pop

  subroutine roctx_mark(message)
    character(len=*), intent(in) :: message
    call roctxMarkA(trim(message)//c_null_char)
  end subroutine roctx_mark

end module roctx_mod


program roctx_demo
  use roctx_mod
  implicit none

  integer, parameter :: N = 1000000
  real(8), allocatable :: A(:), B(:), C(:)
  real(8) :: sum_result
  integer :: i, ret

  print *, "ROCTx Markers Demo (standalone version)"
  print *, "========================================"
  print *, ""

  ret = roctx_range_push("initialization")

  allocate(A(N), B(N), C(N))
  do i = 1, N
    A(i) = real(i, 8)
    B(i) = real(N - i + 1, 8)
  end do
  C = 0.0d0

  ret = roctx_range_pop()

  print *, "Initialized arrays of size:", N

  ret = roctx_range_push("computation")
  call roctx_mark("starting vector add")

  do i = 1, N
    C(i) = A(i) + B(i)
  end do

  call roctx_mark("vector add complete")

  print *, "Computed vector addition"

  ret = roctx_range_push("reduction")

  sum_result = 0.0d0
  do i = 1, N
    sum_result = sum_result + C(i)
  end do

  ret = roctx_range_pop()
  ret = roctx_range_pop()

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
