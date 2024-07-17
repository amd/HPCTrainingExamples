#!/bin/bash

module load mvapich2

mkdir mvapich2_hello_world_f_compile && cd mvapich2_hello_world_f_compile

cat <<-EOF > mpi_hello_world.f
	program mpi_hello_world
        include 'mpi.h'

	integer ierror, size, rank

        // Initialize the MPI environment
        call MPI_Init(ierror);

        // Get the number of processes
        call MPI_Comm_size(MPI_COMM_WORLD, size, ierror);

        // Get the rank of the process
        call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierror);

        // Print off a hello world message
        print *,"Hello world from rank ",rank, " out of ", size, " processors"

        // Finalize the MPI environment.
        call MPI_Finalize();
        }
EOF

mpifort -o mpi_hello_world mpi_hello_world.f
if [ -x mpi_hello_world ]; then
  echo "Executable created"
fi

cd ..
rm -rf mvapich2_hello_world_f_compile
