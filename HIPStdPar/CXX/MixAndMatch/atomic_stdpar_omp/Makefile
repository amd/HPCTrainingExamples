# Compiler and flags
HIPCC        := hipcc
CXXFLAGS     := -std=c++20
OPENMP_FLAGS := -fopenmp
STDPAR_FLAGS := --hipstdpar
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
ARCH_FLAGS   := --offload-arch=${ROCM_GPU}

# If STDPAR_PATH is not specified in the environment, use this default.
STDPAR_PATH ?= ${ROCM_PATH}/include/thrust/system/hip/hipstdpar

# Targets
TARGET       := final
OBJS         := StdParExecutor.o OpenMPExecutor.o

all: $(TARGET)

# Compile OpenMPExecutor.cpp
OpenMPExecutor.o: OpenMPExecutor.cpp
	$(HIPCC) $(OPENMP_FLAGS) -c $< -o $@

# Compile stdpar_executor.cpp
StdParExecutor.o: StdParExecutor.cpp
	$(HIPCC) $(CXXFLAGS) $(STDPAR_FLAGS) --hipstdpar-path=$(STDPAR_PATH) $(ARCH_FLAGS) -c $< -o $@

# Link all object files
$(TARGET): $(OBJS) main.cpp
	$(HIPCC) $(CXXFLAGS) $(OPENMP_FLAGS) $(OBJS) main.cpp -o $@

clean:
	rm -f $(OBJS) $(TARGET)
