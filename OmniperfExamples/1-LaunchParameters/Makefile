HIP_PATH?= $(wildcard /opt/rocm)
ifeq (,$(HIP_PATH))
	HIP_PATH=/opt/rocm
endif

HIPCC=$(HIP_PATH)/bin/hipcc

TARGET=hcc

SOURCES = problem.cpp 
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./problem.exe

.PHONY: test


all: $(EXECUTABLE)

CXXFLAGS = -O3

CXX=$(HIPCC)


$(EXECUTABLE): $(OBJECTS) 
	$(HIPCC) $(OBJECTS) -o $@


test: $(EXECUTABLE)
	$(EXECUTABLE)


clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)


