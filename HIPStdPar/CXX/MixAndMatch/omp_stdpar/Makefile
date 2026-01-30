# Compiler and flags
HIPCC        := hipcc
CXXFLAGS     := -std=c++20
OPENMP_FLAGS := -fopenmp
STDPAR_FLAGS := --hipstdpar
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
ARCH_FLAGS   := --offload-arch=${ROCM_GPU}

# If STDPAR_PATH is not specified on the command line, use this default.
STDPAR_PATH ?= ${ROCM_PATH}/include/thrust/system/hip/hipstdpar

# Targets
TARGET       := final
OBJS         := openmp_executor.o stdpar_executor.o

all: $(TARGET)

# Compile openmp_executor.cpp
openmp_executor.o: openmp_executor.cpp
	$(HIPCC) $(OPENMP_FLAGS) -c $< -o $@

# Compile stdpar_executor.cpp
stdpar_executor.o: stdpar_executor.cpp
	$(HIPCC) $(CXXFLAGS) $(STDPAR_FLAGS) --hipstdpar-path=$(STDPAR_PATH) $(ARCH_FLAGS) -c $< -o $@

# Link all object files and main.cpp into final
$(TARGET): $(OBJS) main.cpp
	$(HIPCC) $(CXXFLAGS) $(OPENMP_FLAGS) $(OBJS) main.cpp -o $@

clean:
	rm -f $(OBJS) $(TARGET)
