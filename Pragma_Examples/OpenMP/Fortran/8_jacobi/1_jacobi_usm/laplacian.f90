! Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
! This software is distributed under the MIT License
!
module laplacian_mod
  use kind_mod
  use mesh_mod, only: mesh_t
  implicit none
  !$omp requires unified_shared_memory

  private
  public :: laplacian, fused_laplacian_update

contains

  subroutine laplacian(mesh,u,au)
    type(mesh_t), intent(inout) :: mesh
    real(RK), intent(inout) :: u(:,:)
    real(RK), intent(inout) :: au(:,:)
    integer(IK) :: i,j
    real(RK) :: invdx2,invdy2

    invdx2 = mesh%dx**-2
    invdy2 = mesh%dy**-2

    !$omp target teams distribute parallel do collapse(2)
    do j = 2,mesh%n_y-1
      do i = 2,mesh%n_x-1
        au(i,j) = (-u(i-1,j)+2._RK*u(i,j)-u(i+1,j))*invdx2 &
                + (-u(i,j-1)+2._RK*u(i,j)-u(i,j+1))*invdy2
      end do
    end do

  end subroutine laplacian

  subroutine fused_laplacian_update(mesh,rhs,u,au,res)
    type(mesh_t), intent(inout) :: mesh
    real(RK), intent(inout) :: rhs(:,:), u(:,:), au(:,:), res(:,:)
    integer(IK) :: i,j
    real(RK) :: invdx2, invdy2, factor, temp

    invdx2 = mesh%dx**-2
    invdy2 = mesh%dy**-2
    factor = (2._RK/mesh%dx**2 + 2._RK/mesh%dy**2)**-1

    !$omp target teams distribute parallel do collapse(2) thread_limit(256)
    do j = 2, mesh%n_y-1
      do i = 2, mesh%n_x-1
        ! Laplacian: compute au(i,j)
        au(i,j) = (-u(i-1,j)+2._RK*u(i,j)-u(i+1,j))*invdx2 &
                + (-u(i,j-1)+2._RK*u(i,j)-u(i,j+1))*invdy2
        
        ! Update: use au(i,j) immediately
        temp = rhs(i,j) - au(i,j)
        res(i,j) = temp
        u(i,j) = u(i,j) + temp*factor
      end do
    end do

  end subroutine fused_laplacian_update

end module laplacian_mod
