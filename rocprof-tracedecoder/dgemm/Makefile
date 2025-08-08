EXECUTABLE = ./dgemm
all: $(EXECUTABLE)

.PHONY: test

OBJECTS = dgemm.o

CXXFLAGS = -g -O2 -DNDEBUG -fPIC
HIPCC_FLAGS = -g -O2 -DNDEBUG
LIBS = -L${ROCM_PATH}/lib -lrocblas

HIP_PLATFORM ?= amd

ifeq ($(HIP_PLATFORM), nvidia)
   HIP_PATH ?= $(shell hipconfig --path)
   HIPCC_FLAGS += -x cu -I${HIP_PATH}/include/
endif
ifeq ($(HIP_PLATFORM), amd)
   HIPCC_FLAGS += -x hip -munsafe-fp-atomics
endif

%.o: %.hip
	hipcc $(HIPCC_FLAGS) -c $^ -o $@

$(EXECUTABLE): $(OBJECTS)
	hipcc $< $(LIBS) $(LDFLAGS) -o $@

test: $(EXECUTABLE)
	rocprofv3 --att -d out -- ./$(EXECUTABLE)

clean:
	rm -rf $(EXECUTABLE) $(OBJECTS) build
