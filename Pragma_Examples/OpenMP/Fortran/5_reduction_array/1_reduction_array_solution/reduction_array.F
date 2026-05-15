! Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
! This software is distributed under the MIT License
!
! This program is based on a reproducer created by Mahdieh Ghazimirsaeed
!
! ======================================================================
! Array reductions in OpenMP target offload (Fortran)
! ======================================================================
!
! OpenMP 5.0+ permits three kinds of list items in a reduction clause:
!
!   1. Whole array      :  reduction(+:ce)
!   2. Array section    :  reduction(+:ce(1:2))
!   3. Array element(s) :  reduction(+:ce(1), ce(2))
!
! All three are standard-conforming.  An array element is treated as a
! length-1 array section (OpenMP 5.2, section 5.5.10).  Listing
! non-overlapping elements in the same reduction clause is permitted;
! only overlapping array sections are forbidden.
!
! In practice, support varies between compilers:
!
!   - Whole-array form is the most portable (gfortran, nvhpc, Cray ftn,
!     classic flang, ifx/ifort, amdflang).  Prefer this form when you
!     can.
!   - Array-section form is supported by most modern OpenMP compilers
!     but historically has had device-runtime gaps.
!   - Array-element list form is rarely exercised and is the form most
!     likely to expose compiler front-end bugs.  For example, amdflang
!     22.0.0 rejects it with
!       'omp.wsloop' op accumulator variable used more than once
!     because both elements alias the same array variable internally.
!
! For instructional purposes the two less-portable forms are kept in
! the source under -DFUTURE.  The default build only exercises the
! whole-array form, which is what the test harness expects to pass.
!
! Build with the experimental forms enabled:
!   amdflang -DFUTURE -fopenmp --offload-arch=<gfx>           \
!            -ffree-form reduction_array.F -o reduction_array
! ======================================================================

    program reduction
    implicit none

    integer, parameter :: real8 = kind(0.0d0)
    real(real8) :: ce(2)
    integer :: j

    ! ------------------------------------------------------------------
    ! Form 1 (recommended, portable): whole-array reduction.
    ! ------------------------------------------------------------------
    ce = 0.0_real8

    ! NOTE: map(tofrom:ce) is REDUNDANT per the OpenMP 5.2 standard.
    ! - Non-scalars referenced in a target region are implicitly map(tofrom)
    !   by the default data-mapping rules (OpenMP 5.2 sec. 5.8.3).
    ! - A reduction list item on a target construct is also implicitly mapped
    !   so the final reduced value is returned to the original variable
    !   (OpenMP 5.2 sec. 5.5.10 / sec. 13.8).
    ! It is kept here as a workaround for an amdflang-new gap (ROCm 7.2.x):
    ! the implicit map for a whole-array reduction(+:ce) on a combined
    ! target teams distribute parallel do is not synthesised, so ce is
    ! captured by host-pointer reference and the device page-faults
    ! ("Memory access fault ... Reason: Page not present") on a non-USM
    ! (xnack-off) build. Remove once amdflang-new emits the implicit map.
    !$omp target teams distribute parallel do reduction(+:ce) map(tofrom:ce)
    do j = 1, 1000
        ce(1) = ce(1) + 1.0_real8
        ce(2) = ce(2) + 1.0_real8
    end do
    !$omp end target teams distribute parallel do

    write(*,'("Form 1 (whole array)   : ce(1) = ",f17.12, &
             &"  ce(2) = ",f17.12)') ce(1), ce(2)

#ifdef FUTURE
    ! ------------------------------------------------------------------
    ! Form 2 (OpenMP 4.5+): array-section reduction.  Standard, equivalent
    ! to Form 1 here, but expresses the section explicitly.  Some older
    ! device runtimes have lacked the helper symbols this lowers to.
    ! ------------------------------------------------------------------
    ce = 0.0_real8

    !$omp target teams distribute parallel do reduction(+:ce(1:2))
    do j = 1, 1000
        ce(1) = ce(1) + 1.0_real8
        ce(2) = ce(2) + 1.0_real8
    end do
    !$omp end target teams distribute parallel do

    write(*,'("Form 2 (array section) : ce(1) = ",f17.12, &
             &"  ce(2) = ",f17.12)') ce(1), ce(2)

    ! ------------------------------------------------------------------
    ! Form 3 (OpenMP 5.0+): array-element list reduction.  Standard-
    ! conforming but historically the most fragile across compilers;
    ! amdflang 22.0.0 currently rejects this with
    !   'omp.wsloop' op accumulator variable used more than once
    ! because the front end sees both elements as references to the same
    ! storage.  Kept here so the example documents the form and can be
    ! used to verify when a compiler has fixed the bug.
    ! ------------------------------------------------------------------
    ce = 0.0_real8

    !$omp target teams distribute parallel do reduction(+:ce(1), ce(2))
    do j = 1, 1000
        ce(1) = ce(1) + 1.0_real8
        ce(2) = ce(2) + 1.0_real8
    end do
    !$omp end target teams distribute parallel do

    write(*,'("Form 3 (element list)  : ce(1) = ",f17.12, &
             &"  ce(2) = ",f17.12)') ce(1), ce(2)
#endif

    end program reduction
