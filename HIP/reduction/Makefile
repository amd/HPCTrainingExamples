# ------------------------------

HIPCC    = hipcc
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
HIPFLAGS = --offload-arch=${ROCM_GPU} 

# ------------------------------

all: reduction_striding reduction_no_striding reduction_to_fix reduction_atomic reduction_two_kernels

reduction_striding: reduction_striding.cpp
	${HIPCC} ${HIPFLAGS} reduction_striding.cpp -o reduction_striding

reduction_no_striding: reduction_no_striding.cpp
	${HIPCC} ${HIPFLAGS} reduction_no_striding.cpp -o reduction_no_striding

reduction_to_fix: reduction_to_fix.cpp
	${HIPCC} ${HIPFLAGS} reduction_to_fix.cpp -o reduction_to_fix

reduction_atomic: reduction_atomic.cpp
	${HIPCC} ${HIPFLAGS} reduction_atomic.cpp -o reduction_atomic

reduction_two_kernels: reduction_two_kernels.cpp
	${HIPCC} ${HIPFLAGS} reduction_two_kernels.cpp -o reduction_two_kernels

.PHONY: clean

clean:
	rm -f reduction_striding reduction_no_striding reduction_to_fix reduction_atomic reduction_two_kernels
