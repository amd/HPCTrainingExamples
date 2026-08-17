# Makefile — MPO Training: FFTs on MI300A
#
#   make            # build all binaries
#   make pdf        # render the exercise markdown to PDF (needs pandoc + LaTeX)
#   make clean

HIPCC   ?= hipcc
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
HIPFLAGS ?= -g -O3 --offload-arch=$(ROCM_GPU)

PYTHON  ?= python3
PANDOC  ?= pandoc
NCOUNT  ?= 5000

HEFFTE_ROOT ?= $(HOME)/heffte-rocm

DOC     = hip_fft_demo_MI300A.md
PDF     = hip_fft_demo_MI300A.pdf
BINS = rocfft_c2c rocfft_3d hipfft_c2c fftw_c2c

.PHONY: all fft heffte pdf clean
all: $(BINS) names.txt

rocfft_c2c: rocfft_c2c.hip
	$(HIPCC) $(HIPFLAGS) $< -lrocfft -o $@

rocfft_3d: rocfft_3d.hip
	$(HIPCC) $(HIPFLAGS) $< -lrocfft -o $@

hipfft_c2c: hipfft_c2c.hip
	$(HIPCC) $(HIPFLAGS) $< -lhipfft -o $@

# FFTW3 drop-in: plain FFTW3 source dispatched to the GPU by hipFFTW (ROCm 7.2.4+).
# The hipFFTW header includes C++ standard headers, so compile the TU as C++ (-x c++).
fftw_c2c: fftw_c2c.c
	$(HIPCC) $(HIPFLAGS) -x c++ $< -lhipfftw -o $@

# Native distributed FFT: MPI-enabled hipFFT (ROCm 7.2.4+), no external solver.
hipfftmp_3d: hipfftmp_3d.cpp
	$(HIPCC) $(HIPFLAGS) $< ${MPI_CXXFLAGS} ${MPI_LDFLAGS} -lmpi -lhipfft -o $@

# Distributed FFT: requires a heFFTe build (see hip_fft_demo_MI300A.md sec 5.1)
heffte_3d: heffte_3d.cpp
	$(HIPCC) $(HIPFLAGS) $< -I$(HEFFTE_ROOT)/include ${MPI_CXXFLAGS} \
	  -L$(HEFFTE_ROOT)/lib ${MPI_LDFLAGS} -lheffte -lrocfft -o $@

pdf: $(PDF)
$(PDF): $(DOC)
	$(PANDOC) $(DOC) -o $(PDF) --toc --highlight-style=tango \
	  -V colorlinks=true -V geometry:margin=1in

clean:
	rm -f $(BINS) hipfftmp_3d heffte_3d
