EXECUTABLE = ./vectoradd
all: $(EXECUTABLE) test

.PHONY: test

OBJECTS = vectoradd.o

CXXFLAGS = -g -O2 -DNDEBUG
HIPCC_FLAGS = -O2 -g -DNDEBUG

HIPCC ?= hipcc

ifeq ($(HIPCC), nvcc)
   HIPCC_FLAGS += -x cu
   LDFLAGS = -lcudadevrt -lcudart_static -lrt -lpthread -ldl
endif
ifeq ($(HIPCC), hipcc)
   HIPCC_FLAGS += -munsafe-fp-atomics
   LDFLAGS = -L${ROCM_PATH}/hip/lib -lamdhip64
   ifeq ($(CXX),)
      ifeq (, $(findstring CXX, clang++))
         LDFLAGS = --hip-link
      endif
   endif
endif

%.o: %.hip
	$(HIPCC) $(HIPCC_FLAGS) -c $^ -o $@

$(EXECUTABLE): $(OBJECTS) 
	$(CXX) $< $(LDFLAGS) -o $@

test: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE) $(OBJECTS)
