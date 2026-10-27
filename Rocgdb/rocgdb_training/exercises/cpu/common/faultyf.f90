! Copyright Advanced Micro Devices, Inc.
!
! SPDX-License-Identifier: MIT

program faulty
  implicit none
  integer, pointer :: p
  integer :: x
  nullify(p)              ! p points to NULL
  x = deref(p)
  print '(i0)', x
contains
  integer function deref(p)
    integer, pointer, intent(in) :: p
    deref = p             ! crashes when p is unassociated (NULL)
  end function deref
end program faulty
