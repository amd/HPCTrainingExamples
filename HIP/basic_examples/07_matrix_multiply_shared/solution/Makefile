# ------------------------------

HIPCC    = hipcc
HIPFLAGS = --offload-arch=gfx908,gfx90a

# ------------------------------

matrix_multiply: matrix_multiply.o
	${HIPCC} matrix_multiply.o -o matrix_multiply

matrix_multiply.o: matrix_multiply.cpp
	${HIPCC} -c matrix_multiply.cpp

.PHONY: clean

clean:
	rm -f matrix_multiply *.o
