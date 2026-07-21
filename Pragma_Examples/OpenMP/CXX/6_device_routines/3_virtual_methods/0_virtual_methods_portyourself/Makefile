EXEC = bigscience
default: ${EXEC}
all: ${EXEC}

#ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

#OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)

CXXFLAGS = -g -fstrict-aliasing ${OPENMP_FLAGS}
LDFLAGS = ${OPENMP_FLAGS} -fno-lto -lm

${EXEC}: ${EXEC}.o HotScience_member_functions.o
	$(CXX) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o ${EXEC}
