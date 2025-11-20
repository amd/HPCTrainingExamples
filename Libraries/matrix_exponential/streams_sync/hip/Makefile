# ------------------------------

ROCM_PATH ?= /opt/rocm
HIPCC = ${ROCM_PATH}/llvm/bin/amdclang++
# ifdef CXX and not equals to HIPCC and g++, then set CXX to HIPCC
ifneq ($(CXX), g++)
	ifneq ($(CXX), ${HIPCC})
	HIPCC = ${CXX}
	endif
endif
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
OPENMP_FLAGS = -fopenmp
HIPFLAGS = -I${ROCM_PATH}/include ${OPENMP_FLAGS} -I${ROCM_PATH}/include/rocprofiler-sdk-roctx
LDFLAGS = -L${ROCM_PATH}/lib -lamdhip64 -lrocblas  -lrocprofiler-sdk-roctx ${OPENMP_FLAGS}

# ------------------------------

mat_exp: mat_exp.o
	${HIPCC} ${HIPFLAGS} ${LDFLAGS} mat_exp.o -o mat_exp

mat_exp.o: mat_exp.cpp
	${HIPCC} ${HIPFLAGS} -x hip -c mat_exp.cpp

.PHONY: clean

clean:
	rm -f mat_exp *.o

