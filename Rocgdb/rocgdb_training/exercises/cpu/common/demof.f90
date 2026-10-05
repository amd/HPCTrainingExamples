! Copyright Advanced Micro Devices, Inc.
!
! SPDX-License-Identifier: MIT

program demo
  implicit none
  integer :: data(8) = [3, 1, 4, 1, 5, 9, 2, 6]
  integer :: total, namelen
  character(len=4096) :: name
  integer :: arglen
  character(len=32) :: arg

  call get_command_argument(0, name, arglen)   ! a "library call" analogue (Ch 3)
  namelen = arglen
  total = sum_array(data, 8)                    ! next OVER / step INTO
  print '(a, i0, a, i0)', "sum = ", total, ", name length = ", namelen

  ! Long idle loop for the ATTACH exercise: start it with  ./demof --spin &
  ! in the background, then attach and inspect k.
  if (command_argument_count() >= 1) then
     call get_command_argument(1, arg)
     if (trim(arg) == "--spin") call spin()
  end if

contains

  ! A tiny helper: something to step INTO and to see on the stack.
  integer function add(a, b)
    integer, intent(in) :: a, b
    add = a + b
  end function add

  ! Sum d(1:n).  The loop is the target for break / ignore / watch /
  ! conditional / display and for "until".
  integer function sum_array(d, n)
    integer, intent(in) :: d(*)
    integer, intent(in) :: n
    integer :: i
    sum_array = 0
    do i = 1, n
       sum_array = add(sum_array, d(i))
    end do
    if (sum_array < 0) then          ! never taken - see the "until" exercise
       print '(a)', "overflow!"
    end if
  end function sum_array

  subroutine spin()
    use iso_c_binding, only: c_int
    implicit none
    interface
       subroutine usleep(usec) bind(c, name="usleep")
         import :: c_int
         integer(c_int), value :: usec
       end subroutine usleep
    end interface
    integer :: k
    k = 0
    do
       call usleep(100000_c_int)     ! 0.1 s per tick
       k = k + 1
    end do
  end subroutine spin

end program demo
