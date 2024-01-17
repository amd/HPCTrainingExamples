! This program is based on a reproducer created by Mahdieh Ghazimirsaeed
! Copyright (c) 2024 AMD HPC Application Performance Team
! MIT License
      program reduction
      integer:: i,j,k
      integer, parameter :: real8 = kind(0.0d0)
      real(real8):: ce1,ce2

      implicit none

      ce1=0.0d0
      ce2=0.0d0
!$omp target teams distribute parallel do simd reduction(+:ce1,ce2)
      do j=1, 1000
         ce1=ce1+1
         ce2=ce2+1
      enddo
!$omp end target teams distribute parallel do simd

      write(*,*) "ce1= ", ce1, "ce2= ", ce2
      end program
