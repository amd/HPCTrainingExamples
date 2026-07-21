ifndef SS_LIB_DIR
$(error The Suitesparse lib dir is not set, please set SS_LIB_DIR)
endif
ifndef SS_INCLUDE_DIR
$(error The Suitesparse include dir is not set, please set SS_INCLUDE_DIR)
endif

HIPCC=hipcc
CFLAGS=-I$(SS_INCLUDE_DIR)

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
