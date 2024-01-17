! This program is based on a reproducer created by Mahdieh Ghazimirsaeed
! Copyright (c) 2024 AMD HPC Application Performance Team
! MIT License
      program reduction
      integer:: j
      integer, parameter :: real8 = kind(0.0d0)
      real(real8):: ce(2)

      implicit none

      ce(:)=0.0d0
!$omp target teams distribute parallel do simd reduction(+:ce)
      do j=1, 1000
         ce(1)=ce(1)+1.0d0
         ce(2)=ce(2)+1.0d0
      enddo
!$omp end target teams distribute parallel do simd

      write(*,*) "ce1= ", ce(1), "ce2= ", ce(2)
      end program
