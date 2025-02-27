# ------------------------------

HIPCC    = hipcc
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
HIPFLAGS = --offload-arch=${ROCM_GPU}
LDFLAGS = -L${ROCM_PATH}/lib -lamdhip64 -lrocblas -lrocrand -lrocsparse

# ------------------------------

matrix_exponential: matrix_exponential.o
	${HIPCC} ${HIPFLAGS} matrix_exponential.o -o matrix_exponential

matrix_exponential.o: matrix_exponential.cpp
	${HIPCC} ${HIPFLAGS} -c matrix_exponential.cpp

.PHONY: clean

clean:
	rm -f matrix_exponential *.o
