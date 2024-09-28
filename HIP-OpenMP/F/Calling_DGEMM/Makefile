EXECUTABLE = ./example
all: $(EXECUTABLE) test

.PHONY: test

OBJECTS = main.o rocm_interface.o rocm_calls.o

FCFLAGS = -g -O2
HIPCC_FLAGS = -g -O2 -DNDEBUG -fPIC -I${ROCM_PATH}/include

HIPCC ?= hipcc

ifeq ($(HIPCC), hipcc)
   HIPCC_FLAGS += -munsafe-fp-atomics
   LDFLAGS = -L${ROCM_PATH}/lib -lamdhip64 -lrocblas
endif

main.o: main.F90 rocm_interface.o
	$(FC) $(FC_FLAGS) -c $^ -o $@

rocm_interface.o: rocm_interface.F90
	$(FC) $(FC_FLAGS) -c $^ -o $@

rocm_calls.o: rocm_calls.hip
	$(HIPCC) $(HIPCC_FLAGS) -c $^ -o $@

$(EXECUTABLE): $(OBJECTS)
	$(FC) $(OBJECTS) $(LDFLAGS) -o $@

test: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS) *.mod
