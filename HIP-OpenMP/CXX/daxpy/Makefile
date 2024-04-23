EXECUTABLE = ./daxpy
all: $(EXECUTABLE) test

.PHONY: test

OBJECTS = daxpy.o daxpy_hip.o

CXXFLAGS = -g -O2 -fPIC -D__HOST_CODE__
HIPCC_FLAGS = -g -O2 -DNDEBUG -fPIC -I${ROCM_PATH}/include

HIPCC ?= hipcc

ifeq ($(HIPCC), nvcc)
   HIPCC_FLAGS += -x cu
   LDFLAGS = -lcudadevrt -lcudart_static -lrt -lpthread -ldl
endif
ifeq ($(HIPCC), hipcc)
   HIPCC_FLAGS += -munsafe-fp-atomics -D__DEVICE_CODE__
   LDFLAGS = -L${ROCM_PATH}/lib -lamdhip64
endif

daxpy_hip.hip: daxpy.cc
	cp $^ $@

daxpy_hip.o: daxpy.cc
	$(HIPCC) $(HIPCC_FLAGS) -c $^ -o $@

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

test: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS) *.hip
