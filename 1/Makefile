cc=g++ -O2 -std=c++20 -lpthread -fopenmp 

.PHONY: all 1 2 3 4

all: 1 2 3 4

%.out: %.c Makefile
	$(cc) $< -o $@
%.out: %.cpp Makefile
	$(cc) $< -o $@

4.out: 4.c Makefile
	gcc -Ofast -march=native -std=c11 -lpthread -fopenmp 4.c -o 4.out

1: 1.out
	OMP_NUM_THREADS=10 ./1.out

2: 2.out
	./2.out
	OMP_NUM_THREADS=2 ./2.out
	OMP_NUM_THREADS=4 ./2.out

3: 3.out 3-launcher.out
	./3-launcher.out

4: 4.out
	objdump -d 4.out | less
