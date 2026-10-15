! PROBLEM
!
! `array` is a normal Fortran module-scope array placed in `declare target`
! (default `to`-style). It is therefore permanently in the device data
! environment with infinite reference count, so OpenMP map(to:..) and
! map(from:..) without the `always` modifier silently skip every transfer.
!
! The same trap applies to SAVE-local variables and COMMON-block elements.
!
! Expected (wrong) output:
!     First element:   1.
!     First element:   2.

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
