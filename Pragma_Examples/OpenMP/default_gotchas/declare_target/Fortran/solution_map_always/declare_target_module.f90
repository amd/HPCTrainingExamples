! SOLUTION: map(always, ...)
!
! `array` is still declare target with infinite reference count, but the
! `always` map-type modifier overrides the OpenMP rule that skips the copy
! when the variable is already in the device data environment. The runtime
! now performs the host->device copy on enter and the device->host copy on
! exit.
!
! Works regardless of `HSA_XNACK`.
!
! Expected (correct) output:
!     First element:   2.
!     First element:   4.

module example_mod
    real(kind=8), dimension(100) :: array
    !$omp declare target(array)
end module

program test
    use example_mod
    implicit none
    integer :: i

    array = 1.0d0

    !$omp target enter data map(always, to: array)
    !$omp target teams distribute parallel do
    do i = 1, 100
        array(i) = array(i) * 2.0d0
    enddo
    !$omp end target teams distribute parallel do
    !$omp target exit data map(always, from: array)

    write(*,*) 'First element: ', array(1)

    !$omp parallel do
    do i = 1, 100
        array(i) = array(i) * 2.0d0
    enddo
    !$omp end parallel do

    write(*,*) 'First element: ', array(1)
end program
