# ------------------------------

CXX   = hipcc
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
CXXFLAGS = --offload-arch=${ROCM_GPU}

# ------------------------------

all: transpose_timed transpose_read_contiguous transpose_write_contiguous transpose_tiled transpose_rocblas

transpose_timed.o: transpose_kernels.h
transpose_kernel_read_contiguous.o: transpose_kernels.h
transpose_kernel_write_contiguous.o: transpose_kernels.h
transpose_kernel_tiled.o: transpose_kernels.h

transpose_timed: transpose_timed.o transpose_kernel_read_contiguous.o transpose_kernel_write_contiguous.o  transpose_kernel_tiled.o
	${CXX} ${LDFLAGS} $^ -lrocblas -o transpose_timed

transpose_read_contiguous: transpose_read_contiguous.o transpose_kernel_read_contiguous.o
	${CXX} ${LDFLAGS} $^ -o transpose_read_contiguous

transpose_write_contiguous: transpose_write_contiguous.o transpose_kernel_write_contiguous.o
	${CXX} ${LDFLAGS} $^ -o transpose_write_contiguous

transpose_tiled: transpose_tiled.o transpose_kernel_tiled.o
	${CXX} ${LDFLAGS} $^ -o transpose_tiled

transpose_rocblas: transpose_rocblas.o
	${CXX} ${LDFLAGS} $^ -lrocblas -o transpose_rocblas

.PHONY: clean

clean:
	rm -f *.o transpose_timed transpose_read_contiguous transpose_write_contiguous transpose_tiled transpose_rocblas
	rm -f transpose_kernel_read_contiguous transpose_kernel_write_contiguous transpose_kernel_tiled
