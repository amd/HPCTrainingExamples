# Compiler and flags
HIPCC        := hipcc
CXXFLAGS     := -std=c++20 -O3
STDPAR_FLAGS := --hipstdpar
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
ARCH_FLAGS   := --offload-arch=${ROCM_GPU}
LIBS         ?=
# std::execution (CPU parallel) pulls in TBB; ROCm ships libtbb under rocprofiler-systems.
# If your OS has tbb-devel in the default linker path, you can leave TBB_LIBDIR empty.
TBB_LIBDIR ?= $(shell \
    if [ -f /usr/lib/x86_64-linux-gnu/libtbb.so ]; then echo /usr/lib/x86_64-linux-gnu; \
    elif [ -f $(ROCM_PATH)/lib/rocprofiler-systems/libtbb.so ]; then echo $(ROCM_PATH)/lib/rocprofiler-systems; \
    else echo ""; fi)
ifeq ($(TBB_LIBDIR),)
$(error TBB not found. Install libtbb-dev or set TBB_LIBDIR=<path>)
endif


# If STDPAR_PATH is not specified on the command line, use this default.
STDPAR_PATH ?= ${ROCM_PATH}/include/thrust/system/hip/hipstdpar

# Targets
TARGET       := final
OBJS         := stdpar_cpu_executor.o stdpar_gpu_executor.o

all: $(TARGET)

# Compile stdpar_cpu_executor.cpp 
stdpar_cpu_executor.o: stdpar_cpu_executor.cpp 
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

# Compile stdpar_gpu_executor.cpp 
stdpar_gpu_executor.o: stdpar_gpu_executor.cpp
	$(HIPCC) $(CXXFLAGS) $(STDPAR_FLAGS) --hipstdpar-path=$(STDPAR_PATH) $(ARCH_FLAGS) -c $< -o $@

# Link all object files and main.cpp into final
$(TARGET): $(OBJS) main.cpp
	$(HIPCC) $(CXXFLAGS) $(OPENMP_FLAGS) $(OBJS) main.cpp -o $@ -L$(TBB_LIBDIR) ${LIBS} -ltbb

clean:
	rm -f $(OBJS) $(TARGET)
