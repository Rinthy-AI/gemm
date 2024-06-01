gemm: gemm.cu
	nvcc -m64 -gencode arch=compute_87,code=sm_87 -o gemm gemm.cu
