! Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
! This software is distributed under the MIT License
!
! Acknoledgement:
! Thanks to Simon Clifford for the original reproducer at the hackathon
! in Birmingham, March 2026.

! This example demonstrates wrong results when using common blocks on
! the device WITHOUT the `always` modifier on map clauses.
! The `always` modifier is needed to ensure the data in the common block
! is actually transferred between host and device.
! Without it the runtime may skip the transfer, leading to stale data.

program test
    implicit none
    common/thing/array
    real(kind=8), dimension(100) :: array
    !$omp declare target(/thing/)

    integer i

    array = 1.0d0
    !$omp target enter data map(to:array)
    !$omp target teams distribute parallel do
    do i = 1, 100
        array(i) = array(i) * 2.0d0
    enddo
    !$omp end target teams distribute parallel do
    !$omp target exit data map(from:array)

    call rout(array)

    !$omp parallel do
    do i = 1, 100
        array(i) = array(i) * 2.0d0
    enddo
    !$omp end parallel do

    call rout(array)

end program

subroutine rout(array)
    real(kind=8), dimension(100) :: array
    write(*,*) 'First element: ', array(1)
    write(*,*) 'Last element:  ', array(100)
end subroutine
