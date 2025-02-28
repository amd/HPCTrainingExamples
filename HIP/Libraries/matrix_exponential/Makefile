EXECUTABLE = ./mat_exp
all: $(EXECUTABLE) 

OBJECTS = mat_exp.o mat_exp_hip.o

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
OPENMP_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}

CXXFLAGS = -g -O3 -fPIC -D__HOST_CODE__ ${OPENMP_FLAGS}
HIPCC_FLAGS = -g -O3 -DNDEBUG -fPIC -I${ROCM_PATH}/include 

HIPCC ?= hipcc

HIPCC_FLAGS += -munsafe-fp-atomics -D__DEVICE_CODE__
LDFLAGS = ${OPENMP_FLAGS} -L${ROCM_PATH}/lib -lamdhip64 -lrocblas 

mat_exp_hip.hip: mat_exp.cpp
	cp $^ $@

mat_exp_hip.o: mat_exp.cpp
	$(HIPCC) $(HIPCC_FLAGS) -c $^ -o $@

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS) *.hip
