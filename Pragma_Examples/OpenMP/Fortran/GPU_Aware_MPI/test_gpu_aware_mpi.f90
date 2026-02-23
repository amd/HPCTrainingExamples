program test_gpu_aware_mpi
    use mpi_f08
    use omp_lib
    implicit none

    integer :: rank, nprocs, ierr, len
    character(len=MPI_MAX_PROCESSOR_NAME) :: hostname
    integer :: sendbuf, recvbuf
    integer, parameter :: N = 1024
    real(8), allocatable :: gpu_send(:), gpu_recv(:)
    real(8) :: expected
    integer :: i
    logical :: gpu_test_pass

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    call MPI_Get_processor_name(hostname, len, ierr)

    write(*,'(A,I0,A,I0,A,A)') &
        'Hello from rank ', rank, ' of ', nprocs, ' on ', trim(hostname)

    ! --- Test 1: Basic MPI_Allreduce on CPU ---
    sendbuf = rank
    call MPI_Allreduce(sendbuf, recvbuf, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)

    if (rank == 0) then
        if (recvbuf == nprocs * (nprocs - 1) / 2) then
            write(*,'(A,I0,A)') 'PASS: MPI_Allreduce correct (sum = ', recvbuf, ')'
        else
            write(*,'(A,I0,A,I0)') 'FAIL: MPI_Allreduce expected ', &
                nprocs * (nprocs - 1) / 2, ' got ', recvbuf
        end if
    end if

    ! --- Test 2: GPU-aware MPI with OpenMP target ---
    allocate(gpu_send(N), gpu_recv(N))

    !$omp target enter data map(alloc: gpu_send, gpu_recv)

    ! Initialize data on GPU
    !$omp target teams distribute parallel do
    do i = 1, N
        gpu_send(i) = dble(rank + 1)
        gpu_recv(i) = 0.0d0
    end do

    ! GPU-aware MPI_Allreduce: pass device pointers directly
    !$omp target data use_device_addr(gpu_send, gpu_recv)
    call MPI_Allreduce(gpu_send, gpu_recv, N, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
    !$omp end target data


    ! Verify result on GPU
    expected = dble(nprocs * (nprocs + 1)) / 2.0d0
    gpu_test_pass = .true.
    !$omp target teams distribute parallel do reduction(.and.:gpu_test_pass)
    do i = 1, N
        if (abs(gpu_recv(i) - expected) > 1.0d-10) then
            gpu_test_pass = .false.
        end if
    end do

    !$omp target exit data map(delete: gpu_send, gpu_recv)

    if (rank == 0) then
        if (gpu_test_pass) then
            write(*,'(A)') 'PASS: GPU-aware MPI_Allreduce correct'
        else
            write(*,'(A)') 'FAIL: GPU-aware MPI_Allreduce returned wrong values'
        end if
    end if

    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    if (rank == 0) then
        write(*,'(A)') 'All tests passed.'
    end if

    deallocate(gpu_send, gpu_recv)
    call MPI_Finalize(ierr)
end program test_gpu_aware_mpi
