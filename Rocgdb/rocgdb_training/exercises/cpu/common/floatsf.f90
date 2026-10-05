! Copyright Advanced Micro Devices, Inc.
!
! SPDX-License-Identifier: MIT

program floats
  implicit none
  real(8) :: da(3) = [1.5d0, 2.5d0, 3.5d0]   ! 8 bytes each
  real(4) :: fa(3) = [1.5,   2.5,   3.5]     ! 4 bytes each
  real(8) :: scale = 2.0d0

  print '(a, g0)', "before: scale = ", scale
  scale = scale * 10.0d0                      ! "jump" OVER this line
  print '(a, g0)', "after:  scale = ", scale

  print '(a, 3f5.1, a, 3f5.1)', "da = ", da, "   fa = ", fa
end program floats
