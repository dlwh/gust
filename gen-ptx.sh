PATH="$PATH:/Developer/NVIDIA/CUDA-5.0/bin/"

if [ `uname` == Darwin ]; then
  NVCC_OPTS="-ccbin /usr/bin/llvm-g++-4.2 $NVCC_OPTS"
fi

nvcc $NVCC_OPTS -D TYPE=float -arch sm_21 --ptx src/main/resources/gust/linalg/cuda/map_kernels_float.cu -o src/main/resources/gust/linalg/cuda/map_kernels_float.ptx
nvcc $NVCC_OPTS -D TYPE=double -arch sm_21 --ptx src/main/resources/gust/linalg/cuda/map_kernels_float.cu -o src/main/resources/gust/linalg/cuda/map_kernels_double.ptx

