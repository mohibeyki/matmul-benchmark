CXX=g++
CXXFLAGS= -fopenmp -O2
NVCC=nvcc
NVCCFLAGS= -Wno-deprecated-gpu-targets -Xcompiler -fpic -O2

all: run

run: cuda omp
	./gsmm_cuda 1024 1024 1024
	./gdmm_cuda 1024 1024 1024
	./gsmm_cuda 1048576 32 256
	./gdmm_cuda 1048576 32 256
	./gsmm 1024 1024 1024
	./gdmm 1024 1024 1024
	./gsmm 1048576 32 256
	./gdmm 1048576 32 256

dual: NVCCFLAGS += -DDEVICE_COUNT=2
dual: run

cuda: gdmm_cuda.cu gsmm_cuda.cu
	$(NVCC) $(NVCCFLAGS) gdmm_cuda.cu -o gdmm_cuda
	$(NVCC) $(NVCCFLAGS) gsmm_cuda.cu -o gsmm_cuda

omp: gdmm.cpp gsmm.cpp
	$(CXX) $(CXXFLAGS) gdmm.cpp -o gdmm
	$(CXX) $(CXXFLAGS) gsmm.cpp -o gsmm
