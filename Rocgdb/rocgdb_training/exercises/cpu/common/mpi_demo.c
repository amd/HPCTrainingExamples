// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int value = rank * rank;   /* a per-rank value to break on and inspect */
    printf("rank %d of %d: value = %d\n", rank, size, value);

    MPI_Finalize();
    return 0;
}
