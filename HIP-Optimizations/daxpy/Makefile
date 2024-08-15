SOURCES = daxpy_1.hip daxpy_2.hip daxpy_3.hip daxpy_4.hip daxpy_5.hip
OBJECTS = $(SOURCES:.hip=.o) 
EXECUTABLE = daxpy_1 daxpy_2 daxpy_3 daxpy_4 daxpy_5

CXX=hipcc
LDFLAGS=-L${ROCM_PATH}/lib -lroctx64
CXXFLAGS= -std=c++17 -Wno-unused-result -O2 -g --save-temps -I${ROCM_PATH}/include/roctracer #-Rpass-analysis=kernel-resource-usage

all: ${EXECUTABLE} 

%.o: %.hip
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(EXECUTABLE): %: %.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) 

clean: 	
	rm -f $(EXECUTABLE) 	
	rm -f $(OBJECTS) *.o
	rm -f *amdgcn* *x86_64* results*

