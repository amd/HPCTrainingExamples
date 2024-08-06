# ------------------------------

HIPCC    = hipcc
HIPFLAGS = --offload-arch=gfx908,gfx90a

# ------------------------------

vector_addition: vector_addition.o
	${HIPCC} ${HIPFLAGS} vector_addition.o -o vector_addition

vector_addition.o: vector_addition.cu
	${HIPCC} ${HIPFLAGS} -c vector_addition.cu

.PHONY: clean

clean:
	rm -f vector_addition *.o
