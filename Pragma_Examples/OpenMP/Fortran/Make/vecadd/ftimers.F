! Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
! 
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
! 
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
! 
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
      module timers_mod

      use, intrinsic :: iso_c_binding, only : C_LONG_LONG

      type, bind(C) :: timeval
         integer(C_LONG_LONG) :: tv_sec
         integer(C_LONG_LONG) :: tv_usec
      end type

      interface
         subroutine fcpu_timer_start(sec, usec) &
              bind(C,NAME="cpu_timer_start")
            use, intrinsic :: iso_c_binding, only : C_LONG_LONG
            implicit none
            integer(C_LONG_LONG) :: sec, usec
         end subroutine fcpu_timer_start

         function fcpu_timer_stop(tstart_sec, tstart_usec) result(tstop) &
              bind(C,NAME="cpu_timer_stop")
            use, intrinsic :: iso_c_binding, only : C_DOUBLE, C_LONG_LONG
            implicit none
            integer(C_LONG_LONG), value :: tstart_sec, tstart_usec
            real(C_DOUBLE) :: tstop
         end function fcpu_timer_stop

         subroutine fcpu_timer_accumulate(tstart_sec, tstart_usec, taccumulate) &
              bind(C,NAME="cpu_timer_accumulate")
            use, intrinsic :: iso_c_binding, only : C_DOUBLE, C_LONG_LONG
            implicit none
            integer(C_LONG_LONG), value :: tstart_sec, tstart_usec
            real(C_DOUBLE) :: taccumulate
         end subroutine fcpu_timer_accumulate
      end interface

      end module timers_mod
