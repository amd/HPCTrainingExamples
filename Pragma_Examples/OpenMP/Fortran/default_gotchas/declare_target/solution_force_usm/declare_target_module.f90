! SOLUTION ATTEMPT: -fopenmp-force-usm
!
! Source is identical to ../problem. The intent is to fix the bug by injecting
! `!$omp requires unified_shared_memory` into every translation unit via the
! compile flag (see Makefile), and run with `HSA_XNACK=1`. Under USM the
! compiler should emit only a device-side reference pointer for the
! declare-target variable and the device should access host memory directly.
!

module example_mod
    real(kind=8), dimension(100) :: array
    !$omp declare target(array)
end module

program test
    use example_mod
    implicit none
    integer :: i

    array = 1.0d0

    !$omp target enter data map(to:array)
    !$omp target teams distribute parallel do
    do i = 1, 100
        array(i) = array(i) * 2.0d0
    enddo
    !$omp end target teams distribute parallel do
    !$omp target exit data map(from:array)

    write(*,*) 'First element: ', array(1)

    !$omp parallel do
    do i = 1, 100
        array(i) = array(i) * 2.0d0
    enddo
    !$omp end parallel do

    write(*,*) 'First element: ', array(1)
end program
