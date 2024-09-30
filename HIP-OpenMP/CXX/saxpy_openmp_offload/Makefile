EXECUTABLE = ./saxpy_openmp
all: $(EXECUTABLE) test

.PHONY: test

OBJECTS = saxpy_openmp.o

CXXFLAGS = -g -O2 -fPIC -D__HOST_CODE__

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

test: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
