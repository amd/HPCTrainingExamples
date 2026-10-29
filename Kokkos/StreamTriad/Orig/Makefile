# Makefile for StreamTriad (Orig) - serial baseline, no Kokkos
#
# Usage:
#   make            # build ./StreamTriad
#   make run        # run interactively via srun (set JOBID to your salloc job)
#   make clean

ROCM_PATH := /shared/apps/rhel9/rocm-10.1.0

CXX      := $(ROCM_PATH)/llvm/bin/amdclang++

CXXFLAGS := -std=gnu++20 -march=znver1 -mtune=znver1

LDFLAGS  := -march=znver1 -mtune=znver1 \
            -Wl,-rpath,$(ROCM_PATH)/lib:$(ROCM_PATH)/lib/llvm/lib

SRC        := StreamTriad.cc timer.c
OBJ        := StreamTriad.o timer.o
TARGET_BIN := StreamTriad

.PHONY: all run clean

all: $(TARGET_BIN)

# separate pattern rules per source extension: a single combined rule
# ($(OBJ): $(SRC)) would make $< always resolve to the first prerequisite
# for every target, silently compiling timer.o from the wrong source file.
%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET_BIN): $(OBJ)
	$(CXX) $(LDFLAGS) $^ -o $@

run: $(TARGET_BIN)
	./$(TARGET_BIN)

clean:
	rm -f $(TARGET_BIN) $(OBJ)
