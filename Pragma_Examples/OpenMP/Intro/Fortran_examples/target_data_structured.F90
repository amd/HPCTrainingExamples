program main
  implicit none
  call example()
contains

  subroutine example()
    implicit none
    integer, parameter :: N = 100000
    real(4) :: tmp(N), a(N), b(N), c(N)

    !$omp target data map(alloc:tmp) &
    !$omp                map(to:a, b) &
    !$omp                map(tofrom:c)
    call zeros(tmp, N)
    call compute_kernel_1(tmp, a, N) ! Uses target
    call saxpy(2.0, tmp, b, N)
    call compute_kernel_2(tmp, b, N) ! Uses target
    call saxpy(2.0, c, tmp, N)
    !$omp end target data

    print *, "Example program for structured target data region completed successfully"
  end subroutine example

  subroutine zeros(a, n)
    implicit none
    integer, intent(in) :: n
    real(4), intent(out) :: a(n)
    integer :: i
    !$omp target teams distribute parallel do
    do i = 1, n
      a(i) = 0.0
    end do
  end subroutine zeros

  subroutine saxpy(a, y, x, n)
    implicit none
    integer, intent(in) :: n
    real(4), intent(in) :: a
    real(4), intent(inout) :: y(n)
    real(4), intent(in) :: x(n)
    integer :: i
    !$omp target teams distribute parallel do
    do i = 1, n
      y(i) = a * x(i) + y(i)
    end do
  end subroutine saxpy

  subroutine compute_kernel_1(x, y, n)
    implicit none
    integer, intent(in) :: n
    real(4), intent(inout) :: x(n)
    real(4), intent(in) :: y(n)
    ! Placeholder for OpenMP target computations
  end subroutine compute_kernel_1

  subroutine compute_kernel_2(x, y, n)
    implicit none
    integer, intent(in) :: n
    real(4), intent(inout) :: x(n)
    real(4), intent(in) :: y(n)
    ! Placeholder for OpenMP target computations
  end subroutine compute_kernel_2

end program main
