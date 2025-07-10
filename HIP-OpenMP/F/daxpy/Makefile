EXECUTABLE = ./daxpy
OBJECTS = daxpy_kernel.o hip_interface.o main.o

all: $(EXECUTABLE) daxpy

.PHONY: all clean

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
OPENMP_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}

FC_FLAGS = -g -O2 -fPIC ${OPENMP_FLAGS}
HIPCC_FLAGS = -g -O2 -DNDEBUG -fPIC -I${ROCM_PATH}/include

HIPCC ?= hipcc

HIPCC_FLAGS += -munsafe-fp-atomics -D__DEVICE_CODE__
LDFLAGS = ${OPENMP_FLAGS} -L${ROCM_PATH}/lib -lamdhip64 -lomptarget -lhiprtc


main.o: main.F90 hip_interface.o
	$(FC) $(FC_FLAGS) -c $^ -o $@

hip_interface.o: hip_interface.F90
	$(FC) $(FC_FLAGS) -c $^ -o $@

daxpy_kernel.o: daxpy_kernel.cpp
	$(HIPCC) $(HIPCC_FLAGS) -c $^ -o $@

$(EXECUTABLE): $(OBJECTS)
	$(FC) $(FC_FLAGS) $(OBJECTS) $(LDFLAGS) -o $@

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS) *.mod
