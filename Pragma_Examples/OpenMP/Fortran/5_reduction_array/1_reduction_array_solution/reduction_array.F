! This program is based on a reproducer created by Mahdieh Ghazimirsaeed
! Copyright (c) 2025 AMD HPC Application Performance Team
! MIT License

    program reduction
    implicit none

    integer, parameter :: real8 = kind(0.0d0)
    real(real8) :: ce(2)
    integer :: j

    ce = 0.0_real8

    !$omp target teams distribute parallel do reduction(+:ce(1), ce(2))
    do j = 1, 1000
        ce(1) = ce(1) + 1.0_real8
        ce(2) = ce(2) + 1.0_real8
    end do
    !$omp end target teams distribute parallel do

    write(*,*) "ce(1) = ", ce(1), "ce(2) = ", ce(2)

    end program reduction