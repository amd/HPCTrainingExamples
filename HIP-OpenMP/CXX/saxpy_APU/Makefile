EXECUTABLE = ./saxpy
all: $(EXECUTABLE) test

.PHONY: test

OBJECTS = saxpy_openmp.o saxpy_hip.o

CXXFLAGS = -g -O2 -fPIC -D__HOST_CODE__
HIPCC_FLAGS = -g -O2 -DNDEBUG -fPIC -I${ROCM_PATH}/include

HIPCC ?= hipcc

ifeq ($(HIPCC), hipcc)
   HIPCC_FLAGS += -x hip -munsafe-fp-atomics -D__DEVICE_CODE__
   LDFLAGS = -L${ROCM_PATH}/lib -lamdhip64
endif

saxpy_hip.o: saxpy_hip.hip
	$(HIPCC) $(HIPCC_FLAGS) -c $^ -o $@

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

test: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
