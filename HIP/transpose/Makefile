# ------------------------------

CXX   = hipcc
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
CXXFLAGS = --offload-arch=${ROCM_GPU}

# ------------------------------

all: transpose_timed

transpose_timed.o: transpose_kernels.h
transpose_kernel_read_contiguous.o: transpose_kernels.h
transpose_kernel_write_contiguous.o: transpose_kernels.h
transpose_kernel_tiled.o: transpose_kernels.h

transpose_timed: transpose_timed.o transpose_kernel_read_contiguous.o transpose_kernel_write_contiguous.o  transpose_kernel_tiled.o
	${CXX} ${LDFLAGS} $^ -o transpose_timed

.PHONY: clean

clean:
	rm -f *.o transpose_timed transpose_kernel_read_contiguous transpose_kernel_write_contiguous transpose_kernel_tiled

