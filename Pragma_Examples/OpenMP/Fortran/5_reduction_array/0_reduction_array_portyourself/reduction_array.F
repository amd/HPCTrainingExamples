! Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
! This software is distributed under the MIT License
!
! This program is based on a reproducer created by Mahdieh Ghazimirsaeed
    program reduction
    implicit none

    integer, parameter :: real8 = kind(0.0d0)
    real(real8) :: ce(2)
    integer :: j

    ce = 0.0_real8

    do j = 1, 1000
        ce(1) = ce(1) + 1.0_real8
        ce(2) = ce(2) + 1.0_real8
    end do

    write(*,*) "ce(1) = ", ce(1), "ce(2) = ", ce(2)

    end program reduction
