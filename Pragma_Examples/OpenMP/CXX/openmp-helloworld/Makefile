HIPCC ?= hipcc
ROCM_GPU ?= gfx942
CXXFLAGS = -fopenmp --offload-arch=$(ROCM_GPU)

SOURCES = openmp_helloworld.cpp

EXECUTABLE=./openmp_helloworld.exe

.PHONY: test


all: $(EXECUTABLE) test


$(EXECUTABLE):
	$(HIPCC) $(CXXFLAGS) $(SOURCES) -o $@


test: $(EXECUTABLE)
	$(EXECUTABLE)


clean:
	rm -f $(EXECUTABLE) *.o

