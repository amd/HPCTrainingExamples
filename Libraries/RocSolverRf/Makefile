#modify as needed
SS_MAIN=/sw/frontier/spack-envs/cpe24.11-cpu/opt/cce-18.0.1/suite-sparse-7.7.0-ycovwp3quevzlv45jj4glnjgafqskhon
SS_INCLUDE_DIR = $(SS_MAIN)/include
HIPCC=hipcc
CFLAGS=-I$(SS_INCLUDE_DIR)

SS_LIB_DIR =  $(SS_MAIN)/lib64

LIBS=-lrocsparse -lrocblas -lrocsolver -L$(SS_LIB_DIR) -lumfpack -lklu

OBJS = umfpack_example_mod.o matrix_utils.o 
OBJS_KLU = klu_example.o matrix_utils.o 
#Rules
all: umfpack_example klu_example
%.o: %.cpp
	$(HIPCC) -c -o $@ $< $(CFLAGS)

umfpack_example: $(OBJS)
	$(HIPCC) -o $@ $^ $(CFLAGS) $(LIBS)

klu_example: $(OBJS_KLU)
	$(HIPCC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -rf *.o umfpack_example klu_example
