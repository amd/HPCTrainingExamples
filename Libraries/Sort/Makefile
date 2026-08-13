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

DOC     = hip_hash_sort_demo_MI300A.md
PDF     = hip_hash_sort_demo_MI300A.pdf
BINS    = zip_sort name_sort multinode_zip_sort rocprim_tempsize

.PHONY: all pdf clean
all: $(BINS) names.txt

zip_sort: zip_sort.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

name_sort: name_sort.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

multinode_zip_sort: multinode_zip_sort.hip
	$(HIPCC) $(HIPFLAGS) -I$(MPI_INC) -L$(MPI_LIB) $< -lmpi -o $@

# sec 3.5: measures rocPRIM Onesweep temporary-storage growth vs. n
rocprim_tempsize: rocprim_tempsize.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

names.txt: gen_names.py
	$(PYTHON) gen_names.py $(NCOUNT) > names.txt

pdf: $(PDF)
$(PDF): $(DOC)
	$(PANDOC) $(DOC) -o $(PDF) --toc --highlight-style=tango \
	  -V colorlinks=true -V geometry:margin=1in

clean:
	rm -f $(BINS)
