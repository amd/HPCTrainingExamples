#!/bin/bash

module load openmpi

mkdir openmpi_hello_world_run && cd openmpi_hello_world_run

cat <<-EOF > mpi_hello_world.c
        #include <mpi.h>
        #include <stdio.h>

        int main(int argc, char* argv[]) {
        // Initialize the MPI environment
        MPI_Init(&argc, &argv);

        // Get the number of processes
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Get the rank of the process
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // Get the name of the processor
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI_Get_processor_name(processor_name, &name_len);

        // Print off a hello world message
        printf("Hello world from processor %s, rank %d out of %d processors\n",
                processor_name, rank, size);

        // Finalize the MPI environment.
        MPI_Finalize();
        }
EOF

mpicc -o mpi_hello_world mpi_hello_world.c
mpirun -n 2 ./mpi_hello_world

cd ..
rm -rf openmpi_hello_world_run
