# saxpy6 crashes during execution
all: saxpy_cpu saxpy1 saxpy2 saxpy1f saxpy3 saxpy4 saxpy5 saxpy6 saxpy7 saxpy2f \
	target_data_structured target_data_unstructured target_data_update

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

CXX1=$(notdir $(CXX))

ifneq ($(findstring amdclang++,$(CXX1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
else ifneq ($(findstring clang++,$(CXX1)),)
  OPENMP_FLAGS = -fopenmp --offload-arch=$(ROCM_GPU)
else ifneq ($(findstring g++,$(CXX1)),)
  OPENMP_FLAGS = -fopenmp -foffload=-march=$(ROCM_GPU)
else ifneq ($(findstring CC,$(CXX1)),)
  OPENMP_FLAGS = -fopenmp
else
  OPENMP_FLAGS = -fopenmp --foffload=-march=$(ROCM_GPU) -fopt-info-optimized-omp
endif

CXXFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS}

ifneq ($(findstring amdflang,$(FC1)),)
  FREE_FORM_FLAG = -Mfreeform
else ifneq ($(findstring flang,$(FC1)),)
  FREE_FORM_FLAG = -Mfreeform
else ifneq ($(findstring gfortran,$(FC1)),)
  FREE_FORM_FLAG = -ffree-form
endif

ifneq ($(findstring ftn,$(FC1)),)
  FFLAGS = -g -O3 ${OPENMP_FLAGS}
else
  FFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS}
endif

ifeq (${CXX1},g++-13)
   LDFLAGS = ${OPENMP_FLAGS} -fno-lto -lm
else
   LDFLAGS = ${OPENMP_FLAGS} -lm
endif

# From slide 14 -- CPU version
saxpy_cpu.o: saxpy_cpu.cpp
	$(CXX) -c -g -O3 -fstrict-aliasing saxpy_cpu.cpp -o saxpy_cpu.o

saxpy_cpu: saxpy_cpu.o
	$(CXX) $(LDFLAGS) $^ -o $@

# From slide 16 but without map clause
saxpy1: saxpy1.o
	$(CXX) $(LDFLAGS) $^ -o $@

# From slide 16 with map clause
saxpy2: saxpy2.o
	$(CXX) $(LDFLAGS) $^ -o $@

# From slide 17, Fortran example
# .f90 or .F90 extension sets code to free form
# some additional code has been added to time the loop and a test
# is added to avoid compilers from optimizing out the operation
# Capitalization of file extension, .F90, turns on preprocessing
%.o: %.F90
	$(FC) -c $(FFLAGS) $(FREE_FORM_FLAG) $< -o $@

saxpy1f: saxpy1f.o
	$(FC) $(LDFLAGS) $^ -o $@

# From slide 18 with optimized map clause
saxpy3: saxpy3.o
	$(CXX) $(LDFLAGS) $^ -o $@

# From slide 19 using subroutine call with pointers
saxpy4: saxpy4.o
	$(CXX) $(LDFLAGS) $^ -o $@

# From slide 22 adding parallel for simd
# simd is not recognized by AMD and Nvidia, but others such as Intel still use it
# parallel construct is a triplet of "parallel for simd"
saxpy5: saxpy5.o
	$(CXX) $(LDFLAGS) $^ -o $@

# From slide 24 adding parallel for simd
saxpy6: saxpy6.o
	$(CXX) $(LDFLAGS) $^ -o $@

# From slide 25
# adding teams distribute to hardware levels
# hardware levels is a triplet of "target teams distribute"
saxpy7: saxpy7.o
	$(CXX) $(LDFLAGS) $^ -o $@

# From slide 25 -- Fortran example
saxpy2f: saxpy2f.o
	$(FC) $(LDFLAGS) $^ -o $@

# From slide 27
target_data_structured: target_data_structured.o
	$(CXX) $(LDFLAGS) $^ -o $@

# From slide 27
target_data_unstructured: target_data_unstructured.o
	$(CXX) $(LDFLAGS) $^ -o $@

# From slide 30
target_data_update: target_data_update.o
	$(CXX) $(LDFLAGS) $^ -o $@

run: saxpy_cpu saxpy1 saxpy2 saxpy1f saxpy3 saxpy4 saxpy5 saxpy6 saxpy7 saxpy2f \
        target_data_structured target_data_unstructured target_data_update
	./saxpy_cpu
	./saxpy1
	./saxpy2
	./saxpy1f
	./saxpy3
	./saxpy4
	./saxpy5
	#./saxpy6
	./saxpy7
	./saxpy2f
	./target_data_structured
	./target_data_unstructured
	./target_data_update

clean:
	rm -f saxpy_cpu saxpy1 saxpy2 saxpy1f saxpy3 saxpy4 saxpy5 saxpy5a saxpy6 \
	   saxpy7 saxpy2f target_data_structured target_data_unstructured target_data_update *.o
