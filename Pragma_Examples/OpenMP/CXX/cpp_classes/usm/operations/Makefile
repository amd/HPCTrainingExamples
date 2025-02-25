
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
OPENMP_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}

CXXFLAGS = -g -O3  -Wall -std=c++17 ${OPENMP_FLAGS} -I"../daxpy" 
LDFLAGS= ${OPENMP_FLAGS} 

OBJECTS = norm.o operations.o main.o daxpy.o
EXEC = example 

all: $(EXEC)

norm.o: norm.cpp norm.hpp
	$(CXX) $(CXXFLAGS) -c norm.cpp -o norm.o

daxpy.o: ../daxpy/daxpy.cpp ../daxpy/daxpy.hpp
	$(CXX) $(CXXFLAGS) -c ../daxpy/daxpy.cpp -o daxpy.o

operations.o: operations.cpp operations.hpp 
	$(CXX) $(CXXFLAGS) -c operations.cpp -o operations.o

main.o: main.cpp operations.hpp 
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

$(EXEC): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(EXEC) $(OBJECTS)

clean:
	rm -f $(EXEC) $(OBJECTS)

