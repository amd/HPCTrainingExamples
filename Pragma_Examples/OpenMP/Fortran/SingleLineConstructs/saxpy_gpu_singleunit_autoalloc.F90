module saxpymod
   use iso_fortran_env
   use omp_lib
   public :: saxpy
contains

subroutine saxpy(a, n)
   use iso_fortran_env
   implicit none
   integer,intent(in) :: n
   real(kind=real32),intent(in) :: a
   real(kind=real32) :: x(n), y(n)
   integer :: i, j
   real(kind=real64) :: start, finish

   x(:) = 1.0_real32
   y(:) = 2.0_real32

   start = OMP_GET_WTIME()
   !$omp target teams distribute parallel do
   do i=1,n
     y(i) = a * x(i) + y(i)
   end do
   finish = OMP_GET_WTIME()

   write (*, '("Time of kernel: ",f8.6)') finish-start
   write(*,*) "plausibility check:"
   write(*,'("y(1) ",f8.6)') y(1)
   write(*,'("y(n) ",f8.6)') y(n)
end subroutine saxpy

end module saxpymod

program main
   use iso_fortran_env
   use saxpymod, ONLY:saxpy
   implicit none

   integer,parameter :: n = 100000
   real(kind=real32) :: a

   a = 2.0_real32

   call saxpy(a, n)
end program main
