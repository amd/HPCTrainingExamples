program main
  implicit none
  integer, parameter :: N = 100000
  real(4), allocatable :: tmp(:), a(:), b(:), c(:)

  allocate(tmp(N), a(N), b(N), c(N))
  
  !$omp target enter data map(alloc:tmp, a, b, c)
  call compute(N)
  !$omp target exit data map(delete:tmp, a, b, c)

  deallocate(tmp, a, b, c)
  print *, "Example program for unstructured target data region completed successfully"

contains

  subroutine compute(N)
    implicit none
    integer, intent(in) :: N
    call zeros(tmp, N)
    call compute_kernel_1(tmp, a, N) ! Uses target
    call saxpy(2.0, tmp, b, N)
    call compute_kernel_2(tmp, b, N) ! Uses target
    call saxpy(2.0, c, tmp, N)
  end subroutine compute

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
