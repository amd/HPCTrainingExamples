module computation_module
  contains

  function some_computation(x) result(r)
    implicit none
    double precision, intent(in) :: x
    double precision :: r
    r = 2.0d0 * x
  end function some_computation

  function final_computation(x, y) result(r)
    implicit none
    double precision, intent(in) :: x, y
    double precision :: r
    r = x + y
  end function final_computation

end module computation_module

module update_module
  contains

  subroutine update_input_array_on_the_host(x, N)
    implicit none
    integer, intent(in) :: N
    double precision, intent(inout) :: x(N)
    integer :: i
    do i = 1, N
       x(i) = 2.0d0
    end do
  end subroutine update_input_array_on_the_host

end module update_module

program main
  use computation_module
  use update_module
  implicit none

  integer, parameter :: N = 1000000
  real(8), allocatable :: input(:), tmp(:)
  real(8) :: res
  integer :: i

  allocate(tmp(N), input(N))
  input = 1.0d0
  res = 0.0d0

  !$omp target data map(alloc: tmp) map(to: input) map(tofrom: res)

  !$omp target teams distribute parallel do
  do i = 1, N
     tmp(i) = some_computation(input(i))
  end do

  call update_input_array_on_the_host(input, N)

  !$omp target update to(input)
  
  !$omp target teams distribute parallel do reduction(+:res)
  do i = 1, N
    res = res + final_computation(input(i), tmp(i))
  end do

  !$omp end target data

  print *, "Final result:", res

  deallocate(tmp, input)

end program main
