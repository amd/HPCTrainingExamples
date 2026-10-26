! Copyright Advanced Micro Devices, Inc.
!
! SPDX-License-Identifier: MIT

! Built at -O2: breakpoints on the assignments below all bind to ONE address -
! the compiler fuses these lines into a single instruction.  Rebuild at -O0 to
! make each line addressable again.  noinline keeps compute a real frame to
! break in (the C twin uses __attribute__((noinline))).
integer function compute(x)
  implicit none
  !GCC$ ATTRIBUTES noinline :: compute
  integer, intent(in) :: x
  integer :: a, b, c
  a = 7
  a = a + 7
  b = a * 2
  c = b + x
  compute = c
end function compute

program fusion
  implicit none
  integer :: compute
  print '(i0)', compute(command_argument_count())   ! argc keeps the result live
end program fusion
