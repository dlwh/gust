set NVCC_OPTS= -gencode arch=compute_30,code=sm_30

nvcc %NVCC_OPTS% -D TYPE=float -I src/main/resources/gust/linalg/cuda/ -ptx src/main/resources/gust/linalg/cuda/matrix_kernels_float.cu -o src/main/resources/gust/linalg/cuda/matrix_kernels_float.ptx
nvcc %NVCC_OPTS% -D TYPE=double -I src/main/resources/gust/linalg/cuda/ -ptx src/main/resources/gust/linalg/cuda/matrix_kernels_float.cu -o src/main/resources/gust/linalg/cuda/matrix_kernels_double.ptx

nvcc %NVCC_OPTS% -D TYPE=float -I src/main/resources/gust/linalg/cuda/ -ptx src/main/resources/gust/linalg/cuda/vector_kernels_float.cu -o src/main/resources/gust/linalg/cuda/vector_kernels_float.ptx
