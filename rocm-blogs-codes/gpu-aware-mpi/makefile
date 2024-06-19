export OMPI_CXX=hipcc

CC=mpicxx

gpu-aware: gpu-aware.cpp
	$(CC) gpu-aware.cpp -o gpu-aware.o
clean:
	rm -rf *.x *.i *.o *.s *.mod
