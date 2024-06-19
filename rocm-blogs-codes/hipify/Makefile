EXECUTABLE = vectoradd

all: $(EXECUTABLE) test
.PHONY: test

SOURCE = vectorAdd_hip.cpp
CXXFLAGS = -g -O2 -fPIC
HIPCC_FLAGS = -O2 -g
HIP_PLATFORM ?= amd
HIP_PATH ?= $(shell hipconfig --path)

ifeq ($(HIP_PLATFORM), nvidia)
	HIPCC_FLAGS += -x cu -I${HIP_PATH}/include/
	LDFLAGS = -lcudadevrt -lcudart_static -lrt
endif

ifeq ($(HIP_PLATFORM), amd)
   HIPCC_FLAGS += -x hip
   LDFLAGS = -L${ROCM_PATH}/hip/lib -lamdhip64
endif

$(EXECUTABLE):
	hipcc $(HIPCC_FLAGS) $(LDFLAGS) -o $(EXECUTABLE) $(SOURCE)

test:
	./$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)