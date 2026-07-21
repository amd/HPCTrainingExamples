EXECUTABLE = ./test_allocators
all: $(EXECUTABLE)

.PHONY: test_allocators

OBJECTS = test_allocators.o

CXXFLAGS = -g -O2 -DNDEBUG -fPIC
HIPCC_FLAGS = -g -O2 -DNDEBUG
OPENMP_FLAGS = -fopenmp

HIP_PLATFORM ?= amd

ifeq ($(HIP_PLATFORM), nvidia)
   HIP_PATH ?= $(shell hipconfig --path)
   HIPCC_FLAGS += -x cu -I${HIP_PATH}/include/
endif
ifeq ($(HIP_PLATFORM), amd)
   HIPCC_FLAGS += -x hip -munsafe-fp-atomics
endif

%.o: %.hip
	hipcc $(HIPCC_FLAGS) $(OPENMP_FLAGS) -c $^ -o $@

$(EXECUTABLE): $(OBJECTS)
	hipcc $< $(LDFLAGS) $(OPENMP_FLAGS) -o $@

test: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -rf $(EXECUTABLE) $(OBJECTS) build
