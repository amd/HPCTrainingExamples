module hip_interface
  interface
    subroutine hip_function(a, b, n) bind(C, name="hip_function")
      use iso_c_binding, only: c_ptr, c_int
      implicit none
      type(c_ptr), value :: a, b
      integer(c_int), value :: n
    end subroutine hip_function
  end interface
end module hip_interface

program example
  use hip_interface
  use iso_c_binding
  implicit none

  real(c_double), allocatable, target, dimension(:) :: a, b
  integer :: i
  integer, parameter :: N = 1024

  allocate(a(N),b(N))
  a = 1.0
  b = 2.0
  !$omp target enter data map(to:a,b)

  ! Compute element-wise b = a + b
  !$omp target data use_device_addr(a,b)
  call hip_function(c_loc(a),c_loc(b),N)
  !$omp end target data

  ! Verification
  !$omp target update from(b)
  if (b(1) /= 3.0) then
    print *, "Answer should be 3.0"
    stop 1
  else
    print*, "PASS!"
  end if
  
  !$omp target exit data map (delete:a,b)
  deallocate(a)
  deallocate(b)
end program example
