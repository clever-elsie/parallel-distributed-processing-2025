CC=nvcc -O2 -std=c++17 --use_fast_math

.PHONY: all 1 2 3

all: 1 2 3

%.out: %.cu Makefile 0.cuh
	$(CC) $< -o $@

1: 1.out
	./1.out

2: 2.out
	./2.out one_path
	./2.out two_path

3: 3.out
	./3.out