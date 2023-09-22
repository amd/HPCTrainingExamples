EXECUTABLE = ./vectoradd
all: $(EXECUTABLE)

.PHONY: test

OBJECTS = vectoradd.o

CXX=hipcc
CXXFLAGS = -g -O2 -DNDEBUG

HIP_PLATFORM ?= amd

ifeq ($(HIP_PLATFORM), nvidia)
   HIP_PATH ?= $(shell hipconfig --path)
   CXXFLAGS += -x cu -I${HIP_PATH}/include/
endif
ifeq ($(HIP_PLATFORM), amd)
   CXXFLAGS += -x hip -munsafe-fp-atomics
endif

$(EXECUTABLE): $(OBJECTS)
	hipcc $< $(LDFLAGS) -o $@

test: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -rf $(EXECUTABLE) $(OBJECTS) build
