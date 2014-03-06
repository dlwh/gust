PATH="$PATH:/Developer/NVIDIA/CUDA-5.0/bin/"

if [ `uname` == Darwin ]; then
  NVCC_OPTS="-gencode;arch=compute_30,code=sm_30 -ccbin /usr/bin/llvm-g++-4.2 $NVCC_OPTS"
fi

nvcc $NVCC_OPTS -D TYPE=float -arch sm_21 --ptx src/main/resources/snap/linalg/cuda/map_kernels_float.cu -o src/main/resources/snap/linalg/cuda/map_kernels_float.ptx
nvcc $NVCC_OPTS -D TYPE=double -arch sm_21 --ptx src/main/resources/snap/linalg/cuda/map_kernels_float.cu -o src/main/resources/snap/linalg/cuda/map_kernels_double.ptx

