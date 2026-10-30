# Makefile for Kokkos StreamTriad (Ver1) targeting AMD MI300A (gfx942)
#
# Usage:
#   make            # build ./StreamTriad
#   make run        # run interactively via srun (set JOBID to your salloc job)
#   make clean

ROCM_PATH   ?= /shared/apps/rhel9/rocm-10.1.0
Kokkos_DIR  ?= ${HOME}/install/Kokkos_HIP

CXX      := $(ROCM_PATH)/llvm/bin/amdclang++
# Set your target AMD GPU architecture
# MI250x : gfx90a
# MI355x/MI350 : gfx950
# MI300A/MI300x : gfx942
TARGET   := gfx942,gfx90a,gfx950

CXXFLAGS := -std=gnu++20 -march=znver1 -mtune=znver1 -fno-gpu-rdc \
            -xhip --rocm-path=$(ROCM_PATH) --offload-arch=$(TARGET) \
            -x hip -fopenmp=libomp \
            -DKOKKOS_DEPENDENCE -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1 \
            -isystem $(Kokkos_DIR)/include -isystem $(ROCM_PATH)/include

LDFLAGS  := -march=znver1 -mtune=znver1 -fno-gpu-rdc \
            -Wl,-rpath,$(ROCM_PATH)/lib:$(ROCM_PATH)/lib/llvm/lib

LDLIBS   := $(Kokkos_DIR)/lib64/libkokkoscontainers.a \
            $(Kokkos_DIR)/lib64/libkokkosalgorithms.a \
            $(Kokkos_DIR)/lib64/libkokkossimd.a \
            $(Kokkos_DIR)/lib64/libkokkoscore.a \
            $(ROCM_PATH)/lib/libamdhip64.so.7.16.26385-0000000 \
            --hip-link \
            $(ROCM_PATH)/lib/llvm/lib/clang/24/lib/linux/libclang_rt.builtins-x86_64.a \
            $(ROCM_PATH)/lib/llvm/lib/libomp.so \
            /lib64/libpthread.a \
            -ldl

SRC        := StreamTriad.cc timer.c
OBJ        := StreamTriad.o timer.o
TARGET_BIN := StreamTriad

.PHONY: all run clean

all: $(TARGET_BIN)

# compile and link as separate steps: CXXFLAGS sets "-x hip" which must not
# leak onto the .a/.so arguments of the link step (clang would try to parse
# them as HIP source otherwise).
#
# Separate pattern rules per source extension: a single combined rule
# ($(OBJ): $(SRC)) would make $< always resolve to the first prerequisite
# (StreamTriad.cc) for every target, silently compiling timer.o from the
# wrong source file.
%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET_BIN): $(OBJ)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDLIBS)

run: $(TARGET_BIN)
	HSA_XNACK=1 ./$(TARGET_BIN)

clean:
	rm -f $(TARGET_BIN) $(OBJ)
