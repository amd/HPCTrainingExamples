# ------------------------------

HIPCC    = hipcc
HIPFLAGS = --offload-arch=gfx908,gfx90a

# ------------------------------

vector_addition: vector_addition.o
	${HIPCC} vector_addition.o -o vector_addition

vector_addition.o: vector_addition.cpp
	${HIPCC} -c vector_addition.cpp

.PHONY: clean

clean:
	rm -f vector_addition *.o
