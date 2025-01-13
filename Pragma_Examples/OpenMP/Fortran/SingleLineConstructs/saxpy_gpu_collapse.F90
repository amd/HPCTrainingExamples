module saxpymod
   use iso_fortran_env
   use omp_lib
   public :: saxpy
contains

subroutine saxpy(a, x, y, m, n)
   use iso_fortran_env
   implicit none
   integer,intent(in) :: n
   real(kind=real32),intent(in) :: a
   real(kind=real32), dimension(:,:),allocatable,intent(in) :: x
   real(kind=real32), dimension(:,:),allocatable,intent(inout) :: y
   integer :: i, j
   real(kind=real64) :: start, finish

   start = OMP_GET_WTIME()
   !$omp target teams loop collapse(2)
   do j=1,n
     do i=1,m
       y(i,j) = a * x(i,j) + y(i,j)
     end do
   end do
   finish = OMP_GET_WTIME()

   write (*, '("Time of kernel: ",f8.6)') finish-start
   write(*,*) "plausibility check:"
   write(*,'("y(1) ",f8.6)') y(1)
   write(*,'("y(n-1) ",f8.6)') y(n-1)
end subroutine saxpy

end module saxpymod

program main
   use iso_fortran_env
   use saxpymod, ONLY:saxpy
   implicit none

   integer,parameter :: n = 1000, m = 1000
   real(kind=real32), allocatable, dimension(:,:) :: x, y
   real(kind=real32) :: a
   integer :: i

   allocate(x(1:m,1:n), y(1:m,1:n))
   a = 2.0_real32
   x(:,:) = 1.0_real32
   y(:,:) = 2.0_real32

   call saxpy(a, x, y, m, n)

   deallocate(x,y)
end program main
