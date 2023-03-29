FC               = gfortran
FFLAGS           = -O3 -g
LDFLAGS += -fopenmp
 
EXEC=freduce
 
all: $(EXEC)
 
%.o:%.F90
	$(FC) $(FFLAGS) $(LDFLAGS) -c $< -o $@
 
$(EXEC): freduce.o
	$(FC) -o $@ $^ $(LDFLAGS)
 
clean:
	rm -rf *.o $(EXEC) *.bin
