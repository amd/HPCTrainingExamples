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
program freduce
        implicit none

        integer :: i, j
        real(kind=8)    :: sum1, sum2
        real(kind=8), pointer :: array(:), buffer(:)
        allocate(array(10))
        allocate(buffer(10))
        
        do j=1, 10
           array(j)=1.0
        end do

        sum2=0.0

        do i=1, 10
           sum2=0.0
           sum1=1000.0
           !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD REDUCTION(+:sum2) &
           !$OMP    MAP(TO:array(1:10))
           do j=1, 10
              sum2=sum2+array(j)
           end do

           buffer(i)=sum1+sum2
        end do

        do i=1, 10
            write(*, *) "sum=", buffer(i)
        end do

        deallocate(array, buffer)
end program
