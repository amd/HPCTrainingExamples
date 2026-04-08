! Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
! This software is distributed under the MIT License
!
module module_interface
  implicit none
  private

  public :: myfunc

  interface myfunc
     module procedure module_func_impl
  end interface myfunc
  
  interface
     module subroutine module_func_impl(i)
       integer :: i
     end subroutine module_func_impl
  end interface
  
end module module_interface

  

  
