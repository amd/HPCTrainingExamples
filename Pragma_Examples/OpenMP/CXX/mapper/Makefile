
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
OPENMP_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}

CXXFLAGS = -g -O3  -Wall -std=c++17 ${OPENMP_FLAGS} 
LDFLAGS= ${OPENMP_FLAGS} 

OBJECTS = daxpy.o main.o
EXEC = mapper

all: $(EXEC)

daxpy.o: daxpy.cpp daxpy.hpp
	$(CXX) $(CXXFLAGS) -c daxpy.cpp -o daxpy.o

main.o: main.cpp daxpy.hpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

$(EXEC): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(EXEC) $(OBJECTS)

clean:
	rm -f $(EXEC) $(OBJECTS)

