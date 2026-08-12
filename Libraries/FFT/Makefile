# Makefile — MPO Training: data-aware hash sorting on MI300A
#
#   make            # build all binaries + generate names.txt
#   make pdf        # render the exercise markdown to PDF (needs pandoc + LaTeX)
#   make names.txt  # (re)generate the sample name list
#   make clean

HIPCC   ?= hipcc
OFFLOAD ?= gfx942
HIPFLAGS ?= -O3 --offload-arch=$(OFFLOAD)

# ROCm-aware MPI (set MPI_HOME in your environment, e.g. `module load cray-mpich`)
MPI_HOME ?= /opt/mpi
MPI_INC  ?= $(MPI_HOME)/include
MPI_LIB  ?= $(MPI_HOME)/lib

PYTHON  ?= python3
PANDOC  ?= pandoc
NCOUNT  ?= 5000

HEFFTE_ROOT ?= $(HOME)/heffte-rocm

DOC     = hip_hash_sort_demo_MI300A.md
PDF     = hip_hash_sort_demo_MI300A.pdf
BINS    = zip_sort name_sort multinode_zip_sort
FFTBINS = rocfft_c2c rocfft_3d hipfft_c2c

.PHONY: all fft heffte pdf clean
all: $(BINS) $(FFTBINS) names.txt

fft: $(FFTBINS)

zip_sort: zip_sort.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

name_sort: name_sort.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

multinode_zip_sort: multinode_zip_sort.hip
	$(HIPCC) $(HIPFLAGS) -I$(MPI_INC) -L$(MPI_LIB) $< -lmpi -o $@

rocfft_c2c: rocfft_c2c.hip
	$(HIPCC) $(HIPFLAGS) $< -lrocfft -o $@

rocfft_3d: rocfft_3d.hip
	$(HIPCC) $(HIPFLAGS) $< -lrocfft -o $@

hipfft_c2c: hipfft_c2c.hip
	$(HIPCC) $(HIPFLAGS) $< -lhipfft -o $@

# Distributed FFT: requires a heFFTe build (see hip_fft_demo_MI300A.md sec 4.1)
heffte_3d: heffte_3d.cpp
	$(HIPCC) $(HIPFLAGS) $< -I$(HEFFTE_ROOT)/include -I$(MPI_INC) \
	  -L$(HEFFTE_ROOT)/lib -L$(MPI_LIB) -lheffte -lmpi -lrocfft -o $@

names.txt: gen_names.py
	$(PYTHON) gen_names.py $(NCOUNT) > names.txt

pdf: $(PDF)
$(PDF): $(DOC)
	$(PANDOC) $(DOC) -o $(PDF) --toc --highlight-style=tango \
	  -V colorlinks=true -V geometry:margin=1in

clean:
	rm -f $(BINS) $(FFTBINS) heffte_3d
