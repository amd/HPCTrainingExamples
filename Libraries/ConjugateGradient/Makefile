HIPCC=hipcc
LIBS=-lrocsparse -lrocblas -lrocsolver

OBJS = cg_driver.o matrix_utils.o
all: run_cg 

%.o: %.cpp
	$(HIPCC) -c -o $@ $< $(CFLAGS)

run_cg: $(OBJS)
	$(HIPCC) -o $@ $^ $(LIBS)


.PHONY: clean

clean:
	rm -rf *.o run_cg
