# ------------------------------

HIPCC    = hipcc
HIPFLAGS = --offload-arch=gfx908,gfx90a
HIPLIBS  = -L${ROCM_PATH}/hipblas/lib -lhipblas

# ------------------------------

matrix_multiply: matrix_multiply.o
	${HIPCC} ${HIPFLAGS} ${HIPLIBS} matrix_multiply.o -o matrix_multiply

matrix_multiply.o: matrix_multiply.cpp
	${HIPCC} ${HIPFLAGS} -c matrix_multiply.cpp

.PHONY: clean

clean:
	rm -f matrix_multiply *.o
