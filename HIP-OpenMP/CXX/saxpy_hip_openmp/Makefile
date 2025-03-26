EXECUTABLE = ./saxpy
all: $(EXECUTABLE) test

.PHONY: test

OBJECTS = saxpy_hip.o saxpy_openmp.o

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
OPENMP_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}

CXXFLAGS = -g -O2 -fPIC ${OPENMP_FLAGS}
HIPCC_FLAGS = -g -O2 -DNDEBUG -fPIC -I${ROCM_PATH}/include

HIPCC ?= hipcc

ifeq ($(HIPCC), nvcc)
   HIPCC_FLAGS += -x cu
   LDFLAGS = -lcudadevrt -lcudart_static -lrt -lpthread -ldl
endif
ifeq ($(HIPCC), hipcc)
   HIPCC_FLAGS += -munsafe-fp-atomics
   LDFLAGS = -L${ROCM_PATH}/lib -lamdhip64 ${OPENMP_FLAGS} 
endif

saxpy_hip.o: saxpy_hip.hip
	$(HIPCC) $(HIPCC_FLAGS) -c $^ -o $@

saxpy_openmp.o: saxpy_openmp.cc
	$(CXX) $(CXXFLAGS) ${OPENMP_FLAGS} -c $^ -o $@

$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(HIPCC_FLAGS) $(OBJECTS) $(LDFLAGS) -o $@

test: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
