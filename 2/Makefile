.PHONY: all 1 2 3 4


%.out: %.cpp Makefile
	mpic++ -O0 -std=c++20 -lpthread -fopenmp $< -o $@

1: 1.out 1-launcher.out
	./1-launcher.out

2:
	make -C ../1 3.out 3-launcher.out
	python3 2img.py
	
%.out: %.cu Makefile
	nvcc -O2 -std=c++20 $< -o $@

3: 3.out
	./3.out before
	./3.out after

4: 4.out
	./4.out