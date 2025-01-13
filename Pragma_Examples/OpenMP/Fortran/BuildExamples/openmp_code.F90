program main
   use iso_fortran_env, only: real64
   implicit none
   integer :: i
   integer :: M=100000
   real(kind=real64) :: sum=0.0_real64;
   real(kind=real64), allocatable, dimension(:) :: in_h, out_h

!$omp requires unified_shared_memory

   allocate(in_h(M))
   allocate(out_h(M))

   do i=1,M ! initialize
      in_h(i) = 1.0;
   enddo

!$omp target teams distribute parallel do map(to:in_h) map(from:out_h)
   do i=1,M
      out_h(i) = in_h(i) * 2.0_real64
   enddo

!$omp target teams distribute parallel do reduction(+:sum) map(to:out_h)
   do i=1,M
     sum = sum + out_h(i);
   enddo

   write (*,"('Result is ',f13.6)") sum

   deallocate(in_h)
   deallocate(out_h)
end program
