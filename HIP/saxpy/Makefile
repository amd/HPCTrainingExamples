CXX_SOURCES = ""
HIP_SOURCES=saxpy.hip

OBJECTS = saxpy.o

EXECUTABLE=./saxpy

.PHONY: test


all: $(EXECUTABLE) test

CXXFLAGS =-g -O2
HIPCC_OPTS=-O2 -g -DNDEBUG

HIPCC?=hipcc

ifeq ($(HIPCC), nvcc)
   HIPCC_FLAGS=$(HIPCC_OPTS) -x cu
   LDFLAGS=-lcudadevrt -lcudart_static -lrt -lpthread -ldl
endif
ifeq ($(HIPCC), hipcc)
   HIPCC_FLAGS=$(HIPCC_OPTS) -munsafe-fp-atomics
   ifeq (, $(findstring CXX, clang++))
      LDFLAGS=--hip-link
   else
      LDFLAGS=-L${ROCM_PATH}/hip/lib -lamdhip64
   endif
endif

%.o : %.hip
	$(HIPCC) $(HIPCC_FLAGS) -c $^ -o $@

$(EXECUTABLE): $(OBJECTS) 
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@


test: $(EXECUTABLE)
	$(EXECUTABLE)


clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
