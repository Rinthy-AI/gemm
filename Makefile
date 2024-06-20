gemm: gemm.md
	./convert.py < gemm.md > /tmp/gemm.cu
	nvcc -m64 -gencode arch=compute_87,code=sm_87 -o gemm /tmp/gemm.cu
