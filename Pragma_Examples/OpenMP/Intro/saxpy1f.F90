program main
   use iso_fortran_env
   implicit none
   integer :: n = 1000000
   real(kind=real32) :: a
   real(kind=real32), allocatable, dimension(:) :: x
   real(kind=real32), allocatable, dimension(:) :: y

   allocate(x(1:n))
   allocate(y(1:n))

   x(:) = 1.0_real32
   y(:) = 2.0_real32

   call saxpy(a, x, y, n)
end program main

subroutine saxpy(a, x, y, n)
   use iso_fortran_env
   use omp_lib
   implicit none
   integer :: n, i
   real(kind=real32) :: a
   real(kind=real32), dimension(n) :: x
   real(kind=real32), dimension(n) :: y
   real(kind=real64) :: start, finish

   start = OMP_GET_WTIME()

   !$omp target
   do i=1,n
       y(i) = a * x(i) + y(i)
   end do
   !$omp end target

   finish = OMP_GET_WTIME()
   write (*, '("Time of kernel: ",f8.6)') finish-start

   if (y(i) > 1.0e30) then
      print *,"y(i)",y(i)
   endif
end subroutine
