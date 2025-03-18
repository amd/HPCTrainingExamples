# ------------------------------

HIPCC    = $(CXX)
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
HIPFLAGS = -I${ROCM_PATH}/include ${OPENMP_FLAGS}
LDFLAGS = -L${ROCM_PATH}/lib -lamdhip64 -lrocblas ${OPENMP_FLAGS}

# ------------------------------

mat_exp: mat_exp.o
	${HIPCC} ${HIPFLAGS} ${LDFLAGS} mat_exp.o -o mat_exp

mat_exp.o: mat_exp.cpp
	${HIPCC} ${HIPFLAGS} -x hip -c mat_exp.cpp

.PHONY: clean

clean:
	rm -f mat_exp *.o
