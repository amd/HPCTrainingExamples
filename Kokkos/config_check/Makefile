# Makefile for compiling and running the Kokkos configuration-check example
# with amdclang++ (ROCm's HIP-aware Clang).
#
# Usage:
#   make                     # build
#   make run                 # build (if needed) and run
#   make clean               # remove build artifacts
#
# Override any variable on the command line if your install differs, e.g.:
#   make KOKKOS_PATH=/path/to/Kokkos_HIP OFFLOAD_ARCH=gfx942

# --- Paths and toolchain -----------------------------------------------
KOKKOS_PATH   ?= ${HOME}/install/Kokkos_HIP
CXX           := amdclang++
OFFLOAD_ARCH  ?= gfx90a

# --- Source / target -----------------------------------------------------
SRC    := kokkos_config_check.cpp
TARGET := kokkos_config_check

# --- Compiler / linker flags ----------------------------------------------
# -x hip            : compile as HIP source (required since Kokkos was built
#                      with the HIP backend)
# -std=c++17        : Kokkos's minimum required C++ standard (adjust if your
#                      install was configured with a different standard)
# --offload-arch    : GPU architecture Kokkos was built for (check with
#                      grep -i KOKKOS_ARCH_ $(KOKKOS_PATH)/lib*/cmake/Kokkos/KokkosConfigCommon.cmake)
# -fopenmp          : required because this Kokkos install also has the
#                      OpenMP host backend enabled
# -no-pie -fno-pie  : avoids R_X86_64_32 relocation errors from linking a
#                      non-PIC static libkokkoscore.a into a PIE executable
CXXFLAGS := -x hip -std=c++20 --offload-arch=$(OFFLOAD_ARCH) -fopenmp \
            -fno-pie -I$(KOKKOS_PATH)/include

LDFLAGS  := -no-pie -fopenmp -L$(KOKKOS_PATH)/lib64 -L$(KOKKOS_PATH)/lib
LDLIBS   := -lkokkoscore -lkokkoscontainers

# --- Rules -----------------------------------------------------------------
.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
