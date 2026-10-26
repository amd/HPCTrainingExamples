! Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
! This software is distributed under the MIT License
!
module roctx_mod
  use, intrinsic :: iso_c_binding, only: c_char, c_int
  implicit none

  private
  public :: roctxRangePush, roctxRangePop

  interface
    function roctxRangePush(message) bind(C, name="roctxRangePushA") result(level)
      import :: c_char, c_int
      character(kind=c_char), intent(in) :: message(*)
      integer(c_int) :: level
    end function roctxRangePush

    function roctxRangePop() bind(C, name="roctxRangePop") result(level)
      import :: c_int
      integer(c_int) :: level
    end function roctxRangePop
  end interface

end module roctx_mod
