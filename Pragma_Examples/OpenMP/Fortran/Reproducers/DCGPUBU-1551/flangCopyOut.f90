program flangCopyOut
  !! Entirely self-contained example showing how the amdflang-new
  !! does not compile for a copy-out subourtine example
!  use FLUE, only: RealTraceMultMatMat
   use ISO_C_BINDING, only: C_DOUBLE, C_DOUBLE_COMPLEX
   implicit none !(external)
   integer, parameter :: WP = C_DOUBLE, WC=C_DOUBLE_COMPLEX
   ! set up size
   integer :: NX, NY, NZ, NT
   ! counters
   integer :: nnt, nnx, nny, nnz, ll
   ! some 3x3 complex matrices
   complex(kind=WC), dimension(:,:,:,:,:,:,:), allocatable :: mat1
   complex(kind=WC), dimension(3,3,10) :: mat2
   complex(kind=WC), dimension(:,:,:), allocatable :: mat3
   real(kind=WP) :: tempPL
   ! sum
   real(kind=WP) :: sumTrP
   integer :: nP
   NX = 32
   NY = 32
   NZ = 32
   NT = 8
   nP = 0
   sumTrP = 0.0_WP


   allocate(mat1(NT,NX,NY,NZ,3,3,10))
   allocate(mat3(3,3,10))
   mat1 = (1.0_WP, 0.52_WP)
   mat2 = (2.0_WP, 0.12_WP)
   mat3 = (5.0_WP, 0.25_WP)

   !$omp target teams distribute parallel do collapse(4) default(none) reduction(+:sumTrp,nP) private(ll, tempPL) shared(mat1,mat2,mat3,nx, ny, nz, nt) num_threads(256) thread_limit(256)
   do nnx = 1, nx
      do nny = 1, ny
         do nnz = 1, nz
            do nnt = 1, nt
               do ll=1, 10
                  ! works
                  call RealTraceMultMatMat(tempPL, mat2(:, :, ll), mat2(:, :, ll))
                  sumTrP = sumTrP + 1.0_WP * tempPL
                  call RealTraceMultMatMat(tempPL, mat3(:, :, ll), mat3(:, :, ll))
                  sumTrP = sumTrP + 1.0_WP * tempPL
                  ! does not compile
                  call RealTraceMultMatMat(tempPL, mat1(nnx, nny, nnz, nnt, :, :, ll), mat1(nnx, nny, nnz, nnt, :, :, ll))
                  sumTrP = sumTrP + 1.0_WP * tempPL
                  nP = nP + 1
               end do
            end do
         end do
      end do
   end do

   write (*, *) 'Done', sumTrP, nP

   deallocate(mat1)

 contains

   pure subroutine TraceMultMatMat(TrMM, left, right)
      complex(kind=WC), dimension(3, 3), intent(in) :: left, right
      complex(kind=WC), intent(out) :: TrMM
      !$omp declare target
      !"""
      !# !Takes the trace of (3,3) complex numbers left, right multiplied together
      !Tr(left*right)
      !"""
      TrMM = left(1, 1) * right(1, 1) + left(1, 2) * right(2, 1) + left(1, 3) * right(3, 1) + &
             left(2, 1) * right(1, 2) + left(2, 2) * right(2, 2) + left(2, 3) * right(3, 2) + &
             left(3, 1) * right(1, 3) + left(3, 2) * right(2, 3) + left(3, 3) * right(3, 3)
   end subroutine TraceMultMatMat

   pure subroutine RealTraceMultMatMat(RTrMM, left, right)
      complex(kind=WC), dimension(3, 3), intent(in) :: left, right
      real(kind=WP), intent(out) :: RTrMM
      complex(kind=WC) :: TrMM
      !$omp declare target
      !"""
      !# !Takes the real trace of (3,3) complex numbers left, right multiplied together
      !Real(Tr(left*right))
      !"""
      call TraceMultMatMat(TrMM, left, right)
      RTrMM = real(TrMM, kind=WP)
   end subroutine RealTraceMultMatMat

end program flangCopyOut
