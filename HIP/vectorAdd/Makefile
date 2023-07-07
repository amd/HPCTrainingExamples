CXX_SOURCES = ""
HIP_SOURCES=vectoradd.hip

OBJECTS = vectoradd.o

EXECUTABLE=./vectoradd

.PHONY: test


all: $(EXECUTABLE) test

CXXFLAGS =-g -O2

HIPCC?=hipcc

ifeq ($(HIPCC), nvcc)
   HIPCC_FLAGS=-O2 -g -DNDEBUG -MD -MT -MF -x cu
   LDFLAGS=-lcudadevrt -lcudart_static -lrt -lpthread -ldl
endif
ifeq ($(HIPCC), hipcc)
   HIPCC_FLAGS=-munsafe-fp-atomics
   LDFLAGS=
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
