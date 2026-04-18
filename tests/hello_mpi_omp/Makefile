COMP ?= mpicc
CFLAGS ?= -O2 -fopenmp

hello_mpi_omp: hello_mpi_omp.c
	$(COMP) $(CFLAGS) -o $@ $<

clean:
	rm -f hello_mpi_omp
