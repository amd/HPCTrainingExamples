# Makefile — MPO Training: data-aware hash sorting on MI300A
#
#   make            # build all binaries + generate names.txt
#   make pdf        # render the exercise markdown to PDF (needs pandoc + LaTeX)
#   make names.txt  # (re)generate the sample name list
#   make clean

HIPCC   ?= hipcc
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
HIPFLAGS ?= -O3 --offload-arch=${ROCM_GPU}

# ROCm-aware MPI: auto-detect compile/link flags from mpicxx in PATH
# (OpenMPI supports --showme; override MPI_CFLAGS/MPI_LDFLAGS to customize).
MPICXX      ?= mpicxx
MPI_CFLAGS  ?= $(shell $(MPICXX) --showme:compile 2>/dev/null)
MPI_LDFLAGS ?= $(shell $(MPICXX) --showme:link 2>/dev/null)

PYTHON  ?= python3
PANDOC  ?= pandoc
NCOUNT  ?= 5000

DOC     = hip_hash_sort_demo_MI300A.md
PDF     = hip_hash_sort_demo_MI300A.pdf
BINS    = zip_sort name_sort multinode_zip_sort rocprim_tempsize sort_apis index_sort id_sort email_sort
ECOUNT  ?= 200000

.PHONY: all pdf clean
all: $(BINS) names.txt emails.txt

zip_sort: zip_sort.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

name_sort: name_sort.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

multinode_zip_sort: multinode_zip_sort.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@ -lmpi

# sec 3.5: measures rocPRIM Onesweep temporary-storage growth vs. n
rocprim_tempsize: rocprim_tempsize.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

# library sort APIs side by side (thrust / rocPRIM / hipCUB; header-only)
sort_apis: sort_apis.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

# index sort (argsort) vs. sorting the data itself
index_sort: index_sort.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

# true perfect-hash sort: unique dense keys -> single scatter, no atomics/scan
id_sort: id_sort.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

# emails: unique but sparse keys -> compact hash (load-factor / probe sweep)
email_sort: email_sort.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

names.txt: gen_names.py
	$(PYTHON) gen_names.py $(NCOUNT) > names.txt

emails.txt: gen_emails.py
	$(PYTHON) gen_emails.py $(ECOUNT) > emails.txt

pdf: $(PDF)
$(PDF): $(DOC)
	$(PANDOC) $(DOC) -o $(PDF) --toc --highlight-style=tango \
	  -V colorlinks=true -V geometry:margin=1in

clean:
	rm -f $(BINS)
