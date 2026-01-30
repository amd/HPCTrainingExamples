ROCM_PATH ?=/opt/rocm
SOURCES = daxpy_1.hip daxpy_2.hip daxpy_3.hip daxpy_4.hip daxpy_5.hip
OBJECTS = $(SOURCES:.hip=.o) 
EXECUTABLE = daxpy_1 daxpy_2 daxpy_3 daxpy_4 daxpy_5

HIPCC ?= hipcc
LDFLAGS ?= -L${ROCM_PATH}/lib -lroctx64
HIPCCFLAGS += -std=c++17 -Wno-unused-result -O2 -g --save-temps -I${ROCM_PATH}/include/roctracer -I${ROCM_PATH}/include/rocprofiler-sdk-roctx #-Rpass-analysis=kernel-resource-usage

all: ${EXECUTABLE} 

%.o: %.hip
	$(HIPCC) $(HIPCCFLAGS) -o $@ -c $<

$(EXECUTABLE): %: %.o
	$(HIPCC) $< -o $@ $(LDFLAGS) 

clean: 	
	rm -f $(EXECUTABLE) 	
	rm -f $(OBJECTS) *.o
	rm -f *amdgcn* *x86_64* results*

