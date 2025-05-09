
module saxpymod
   use iso_fortran_env
   use omp_lib
  
   implicit none
   private

   public :: saxpy,initialize

contains
        
subroutine saxpy(a, x, y, n)
   implicit none
   integer,intent(in) :: n
   integer :: i
   real(kind=real32) :: a
   real(kind=real32), dimension(:),allocatable,intent(in) :: x
   real(kind=real32), dimension(:),allocatable,intent(inout) :: y
   real(kind=real64) :: start, finish

   start = OMP_GET_WTIME()
   do i=1,n
       y(i) = a * x(i) + y(i)
   end do
   finish = OMP_GET_WTIME()
   write (*, '("Time of kernel: ",f8.6)') finish-start

   write(*,*) "plausibility check:"
   write(*,'(a,f8.6)') "y(1)",y(1)
   write(*,'(a,f8.6)') "y(n-1)",y(n-1)
end subroutine saxpy

subroutine initialize(x,y,n)
   implicit none 

   integer,intent(in) :: n
   integer :: i
   real(kind=real32), dimension(:),allocatable,intent(inout) :: x
   real(kind=real32), dimension(:),allocatable,intent(inout) :: y

   do i=1,n
     x(i) = 1.0_real32
     y(i) = 2.0_real32
   end do
   end subroutine initialize

end module saxpymod

program main

   use iso_fortran_env
   use saxpymod, ONLY:saxpy,initialize
   implicit none

   integer,parameter :: n = 10000000
   real(kind=real32), allocatable, dimension(:) :: x
   real(kind=real32), allocatable, dimension(:) :: y
   real(kind=real32) :: a

   allocate(x(1:n))
   allocate(y(1:n))

   call initialize(x,y,n)
   a = 2.0_real32

   call saxpy(a, x, y, n)
end program main
