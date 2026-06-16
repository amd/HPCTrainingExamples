EXECUTABLE = ./saxpy_openmp
all: $(EXECUTABLE) test

.PHONY: test

ifdef CRAYPE_VERSION
   OPENMP_FLAGS = -fopenmp
else
   ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
   OPENMP_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}
endif

OBJECTS = saxpy_openmp.o

CXXFLAGS = -g -O2 -fPIC -D__HOST_CODE__ ${OPENMP_FLAGS}
LDFLAGS += ${OPENMP_FLAGS}

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

test: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
