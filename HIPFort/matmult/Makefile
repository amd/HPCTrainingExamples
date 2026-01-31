EXECUTABLE = ./matmult_hipfort
all: $(EXECUTABLE) 

.PHONY: test

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
HIPFORT_PATH ?= ${ROCM_PATH}

OBJECTS = matmult_prog.o matmult_interface.o matmult_hip.o matmult_mod.o

FC_FLAGS = -g -O2 -fopenmp --offload-arch=${ROCM_GPU} -I${HIPFORT_PATH}/include/hipfort/amdgcn
HIPCC_FLAGS = -g -O2 -DNDEBUG -fPIC -I${ROCM_PATH}/include

HIPCC ?= hipcc
HIPCC1=$(notdir $(HIPCC))

ifneq ($(findstring hipcc,$(HIPCC1)),)
   HIPCC_FLAGS += -munsafe-fp-atomics
   LDFLAGS = -L${HIPFORT_PATH}/lib -lhipfort-amdgcn -L${ROCM_PATH}/lib -lhipblas -lamdhip64 -lstdc++
endif

matmult_prog.o: matmult_prog.f90 matmult_interface.o matmult_mod.o
	$(FC) $(FC_FLAGS) -c $^ -o $@

matmult_interface.o: matmult_interface.f90
	$(FC) $(FC_FLAGS) -c $^ -o $@

matmult_mod.o: matmult_mod.f90
	$(FC) $(FC_FLAGS) -c $^ -o $@

matmult_hip.o: matmult_hip.cpp
	$(HIPCC) $(HIPCC_FLAGS) -c $^ -o $@

$(EXECUTABLE): $(OBJECTS)
	$(FC) $(FC_FLAGS) $(OBJECTS) $(LDFLAGS) -o $@

test: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS) *.mod
