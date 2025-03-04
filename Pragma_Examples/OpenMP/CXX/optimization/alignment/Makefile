all: align_64 align_128 align_256 align_512 align_1024

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)

CXXFLAGS = -g -O3 -std=c++17 -fstrict-aliasing ${OPENMP_FLAGS}
LDFLAGS = ${OPENMP_FLAGS} -fno-lto -lm

align_64.o: align.cc
	$(CXX) -c ${CXX_FLAGS} -DBLOCKSIZE=64 $^ -o $@

align_64: align_64.o
	$(CXX) $^ $(LDFLAGS) -o $@

align_128.o: align.cc
	$(CXX) -c ${CXX_FLAGS} -DBLOCKSIZE=128 $^ -o $@

align_128: align_128.o
	$(CXX) $^ $(LDFLAGS) -o $@

align_256.o: align.cc
	$(CXX) -c ${CXX_FLAGS} -DBLOCKSIZE=256 $^ -o $@

align_256: align_256.o
	$(CXX) $^ $(LDFLAGS) -o $@

align_512.o: align.cc
	$(CXX) -c ${CXX_FLAGS} -DBLOCKSIZE=512 $^ -o $@

align_512: align_512.o
	$(CXX) $^ $(LDFLAGS) -o $@

align_1024.o: align.cc
	$(CXX) -c ${CXX_FLAGS} -DBLOCKSIZE=1024 $^ -o $@

align_1024: align_1024.o
	$(CXX) $^ $(LDFLAGS) -o $@

# Cleanup
clean:
	rm -f *.o ${EXEC}
