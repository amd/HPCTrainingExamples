# ------------------------------

HIPCC    = hipcc
HIPFLAGS = --offload-arch=gfx908,gfx90a

# ------------------------------

matrix_addition: matrix_addition.o
	${HIPCC} matrix_addition.o -o matrix_addition

matrix_addition.o: matrix_addition.cpp
	${HIPCC} -c matrix_addition.cpp

.PHONY: clean

clean:
	rm -f matrix_addition *.o
