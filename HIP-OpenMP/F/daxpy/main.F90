program daxpy
  use iso_c_binding
  use hip_interface
  implicit none

  integer, parameter :: n = 1024 ! use 1024 for our example
  real(c_double) :: a = 2.0 ! use 2.0 for our example
  real(c_double), allocatable, target :: x(:), y(:)
  integer :: i

  allocate(x(n), y(n))
  print *, "daxpy Compiled with HOST_CODE"

  ! allocate the device memory
  !$omp target data map(to: x) map(tofrom: y)
  call compute_1(n, x)
  call compute_2(n, y)
  !$omp target update to(x) to(y)

  !$omp target data use_device_addr(x, y)
  call daxpy_hip(n, a, c_loc(x), c_loc(y))
  !$omp end target data

  !$omp end target data

  call compute_3(n, y)

contains

  subroutine compute_1(n, x)
    integer, intent(in) :: n
    real(c_double), dimension(n), intent(out) :: x
    integer :: i
    do i = 1, n
       x(i) = 1.0_c_double
    end do
  end subroutine compute_1

  subroutine compute_2(n, y)
    integer, intent(in) :: n
    real(c_double), dimension(n), intent(out) :: y
    integer :: i
    do i = 1, n
       y(i) = 2.0_c_double
    end do
  end subroutine compute_2

  subroutine compute_3(n, y)
    integer, intent(in) :: n
    real(c_double), dimension(n), intent(in) :: y
    real(c_double) :: total
    integer :: i

    total = 0.0_c_double
    do i = 1, n
       total = total + y(i)
    end do

    if (total == (n * 4.0_c_double)) then
       print *, "PASS: Results are verified as correct."
    else
       print *, "FAIL: Results are not correct. Expected ", n * 4.0_c_double, " and received ", total
    end if
  end subroutine compute_3

end program daxpy
