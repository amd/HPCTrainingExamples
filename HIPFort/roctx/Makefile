# Makefile for ROCTx Fortran Example
# Two build targets:
#   roctx_hipfort    - Uses hipfort's roctx bindings
#   roctx_standalone - Uses custom standalone C bindings (self-contained)

HIPFORT_PATH ?= ${ROCM_PATH}
HIPFORT_INC  ?= ${HIPFORT_PATH}/include/hipfort

FC = amdflang

# Detect roctx library: try roctx64 first, fall back to rocprofiler-sdk-roctx
ROCTX_LIB := $(shell if [ -f "$(ROCM_PATH)/lib/libroctx64.so" ]; then echo "roctx64"; else echo "rocprofiler-sdk-roctx"; fi)
LDFLAGS = -L${ROCM_PATH}/lib -l$(ROCTX_LIB)

all: roctx_hipfort roctx_standalone

roctx_hipfort: roctx_hipfort.F90
	$(FC) -cpp -g -O2 -o $@ $< -I${HIPFORT_INC}/amdgcn $(LDFLAGS)

roctx_standalone: roctx_standalone.F90
	$(FC) -cpp -g -O2 -o $@ $< $(LDFLAGS)

.PHONY: test clean

test: all
	@echo "Testing hipfort version..."
	./roctx_hipfort
	@echo ""
	@echo "Testing standalone version..."
	./roctx_standalone

clean:
	rm -f roctx_hipfort roctx_standalone *.o *.mod
