! SOLUTION: declare target link(...)
!
! `link` keeps `array` out of the device data environment until it is
! explicitly mapped, so the reference count starts at 0 and transitions
! 0->1 on enter and 1->0 on exit. The OpenMP map rules then fire the
! copies on those transitions; the `always` modifier is not needed.
!
! Works regardless of `HSA_XNACK`. With `HSA_XNACK=1` the runtime can
! additionally use auto zero-copy to avoid the physical allocation/copy,
! but correctness does not depend on it.
!
! Expected (correct) output:
!     First element:   2.
!     First element:   4.

module example_mod
    real(kind=8), dimension(100) :: array
    !$omp declare target link(array)
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
