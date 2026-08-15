# Copyright AMD 2026, MIT License
# Author: Bob Robey Bob.Robey@amd.com with AI tool help
#
# Makefile — MPO Training: data-aware hash sorting on MI300A
#
#   make            # build all binaries + generate names.txt
#   make pdf        # render the exercise markdown to PDF (needs pandoc + LaTeX)
#   make names.txt  # (re)generate the sample name list
#   make clean

HIPCC   ?= hipcc
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
HIPFLAGS ?= -g -O3 --offload-arch=${ROCM_GPU}

PYTHON  ?= python3
PANDOC  ?= pandoc
NCOUNT  ?= 5000

# Pick the first available pandoc PDF engine (pdflatex may be absent on training nodes;
# a user-local tectonic in ~/bin is a zero-root fallback).
PDF_ENGINE ?= $(shell for e in pdflatex xelatex lualatex tectonic $$HOME/bin/tectonic; do command -v $$e >/dev/null 2>&1 && { echo $$e; break; }; done)

DOC     = sort_demo_MI300A.md
PDF     = sort_demo_MI300A.pdf
BINS    = zip_sort name_sort multinode_zip_sort rocprim_tempsize sort_apis index_sort id_sort email_sort
ECOUNT  ?= 200000

.PHONY: all pdf clean
all: $(BINS) names.txt emails.txt

# library sort APIs side by side (thrust / rocPRIM / hipCUB; header-only)
sort_apis: sort_apis.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

# index sort (argsort) vs. sorting the data itself
index_sort: index_sort.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

# true perfect-hash sort: unique dense keys -> single scatter, no atomics/scan
id_sort: id_sort.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

zip_sort: zip_sort.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

names.txt: gen_names.py
	$(PYTHON) gen_names.py $(NCOUNT) > names.txt

name_sort: name_sort.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

# emails: unique but sparse keys -> compact hash (load-factor / probe sweep)
emails.txt: gen_emails.py
	$(PYTHON) gen_emails.py $(ECOUNT) > emails.txt

email_sort: email_sort.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

# Scaling to multiple GPUs/nodes
multinode_zip_sort: multinode_zip_sort.hip
	$(HIPCC) $(HIPFLAGS) $(MPI_CXXFLAGS) $< $(MPI_LDFLAGS) -o $@

# measures rocPRIM Onesweep temporary-storage growth vs. n
rocprim_tempsize: rocprim_tempsize.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

pdf: $(PDF)
$(PDF): $(DOC)
	$(PANDOC) $(DOC) -o $(PDF) --toc --highlight-style=tango \
	  -V colorlinks=true -V geometry:margin=1in \
	  $(if $(PDF_ENGINE),--pdf-engine=$(PDF_ENGINE),)

clean:
	rm -f $(BINS) emails.txt names.txt
