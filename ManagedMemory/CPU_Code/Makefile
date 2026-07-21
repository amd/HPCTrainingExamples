all: cpu_code

cpu_code: cpu_code.c
	amdclang -g -O3 cpu_code.c -o cpu_code

clean:
	rm -f cpu_code
