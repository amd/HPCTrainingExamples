HIPCC=hipcc
LIBS=-lrocsparse -lrocblas -lrocsolver

OBJS = cg_driver.o matrix_utils.o preconditioner.o ic_preconditioner.o jacobi_preconditioner.o asynch_jacobi_preconditioner.o pcg.o
all: run_cg 

%.o: %.cpp
	$(HIPCC) -c -o $@ $< $(CFLAGS)

run_cg: $(OBJS)
	$(HIPCC) -o $@ $^ $(LIBS)


.PHONY: clean

clean:
	rm -rf *.o run_cg
