! Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
! This software is distributed under the MIT License
!
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
