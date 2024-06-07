EXECUTABLE = ./problem.exe
all: $(EXECUTABLE)

.PHONY: test

OBJECTS = problem.o
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
CXXFLAGS = -g -O2 -DNDEBUG -fPIC
HIPCC_FLAGS = --offload-arch=$(ROCM_GPU) -g -O2

HIP_PLATFORM ?= amd

ifeq ($(HIP_PLATFORM), nvidia)
   HIP_PATH ?= $(shell hipconfig --path)
   HIPCC_FLAGS += -x cu -I${HIP_PATH}/include/
endif
ifeq ($(HIP_PLATFORM), amd)
   HIPCC_FLAGS += -x hip -munsafe-fp-atomics
endif

%.o: %.cpp
	hipcc $(HIPCC_FLAGS) -c $^ -o $@

$(EXECUTABLE): $(OBJECTS)
	hipcc $< $(LDFLAGS) -o $@

test: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -rf $(EXECUTABLE) $(OBJECTS) build

