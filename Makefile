all:
	$(CUDA_PATH)/bin/nvcc -lcuda -lcublas *.cu -o run  -arch=compute_20
