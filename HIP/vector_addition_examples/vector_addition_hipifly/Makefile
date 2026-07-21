# ------------------------------

HIPCC    = hipcc
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
HIPFLAGS = --offload-arch=${ROCM_GPU}

# ------------------------------

vector_addition: vector_addition.o
	${HIPCC} ${HIPFLAGS} vector_addition.o -o vector_addition

vector_addition.o: vector_addition.cu
	${HIPCC} ${HIPFLAGS} -c vector_addition.cu

.PHONY: clean

clean:
	rm -f vector_addition *.o
