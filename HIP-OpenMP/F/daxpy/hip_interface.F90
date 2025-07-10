module hip_interface
  use iso_c_binding
  implicit none

  interface
    subroutine daxpy_hip(n, a, x, y) bind(C)
      import :: c_int, c_double, c_ptr
      integer(c_int), value :: n
      real(c_double), value :: a
      type(c_ptr), value :: x, y
    end subroutine daxpy_hip
  end interface

end module hip_interface
