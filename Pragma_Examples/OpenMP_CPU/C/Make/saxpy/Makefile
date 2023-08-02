CC               = gcc
CFLAGS           = -O3 -g -std=c99 -fopenmp
LDFLAGS += -fopenmp

EXEC=saxpy 

all: $(EXEC)

$(EXEC): codelet.o main.o
	$(CC) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf *.o $(EXEC)
