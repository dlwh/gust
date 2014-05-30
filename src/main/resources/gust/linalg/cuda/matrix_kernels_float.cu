#include <stdio.h>

#define MAKE_NAME(prefix, fun, T) prefix ## _ ## fun ## _ ## T

#define MAP_FUN_1(fun, T) \
extern "C" \
__global__ void MAKE_NAME(map, fun, T) (int rows, int cols,\
    T *out, int outMajorStride,\
    const T *in, int inMajorStride) {\
  for(int col = threadIdx.x + blockIdx.x * blockDim.x; col < cols; col += blockDim.x * gridDim.x) {\
    for(int row = threadIdx.y + blockIdx.y * blockDim.y; row < rows;  row += blockDim.y * gridDim.y) {\
        out[col * outMajorStride + row] = fun(in[col * inMajorStride + row]);\
    }\
  }\
}



#define MAP_FUN_2(fun, T) \
extern "C" \
__global__ void MAKE_NAME(map2, fun, T) (int rows, int cols,\
    T *out, int outMajorStride,\
    const T *a, int aMajorStride,\
    const T *b, int bMajorStride) {\
  for(int col = threadIdx.x + blockIdx.x * blockDim.x; col < cols; col += blockDim.x * gridDim.x) {\
    for(int row = threadIdx.y + blockIdx.y * blockDim.y; row < rows;  row += blockDim.y * gridDim.y) {\
        out[col * outMajorStride + row] = fun(a[col * aMajorStride + row], b[col * bMajorStride + row]);\
    }\
  }\
}\
\
extern "C" \
__global__ void MAKE_NAME(map2_v_s, fun, T) (int rows, int cols,\
    T *out, int outMajorStride,\
    const T *a, int aMajorStride,\
    const T b) {\
  for(int col = threadIdx.x + blockIdx.x * blockDim.x; col < cols; col += blockDim.x * gridDim.x) {\
    for(int row = threadIdx.y + blockIdx.y * blockDim.y; row < rows;  row += blockDim.y * gridDim.y) {\
        out[col * outMajorStride + row] = fun(a[col * aMajorStride + row], b);\
    }\
  }\
}\
\
extern "C" \
__global__ void MAKE_NAME(map2_s_v, fun, T) (int rows, int cols,\
    T *out, int outMajorStride,\
    const T a,\
    const T *b, int bMajorStride) {\
  for(int col = threadIdx.x + blockIdx.x * blockDim.x; col < cols; col += blockDim.x * gridDim.x) {\
    for(int row = threadIdx.y + blockIdx.y * blockDim.y; row < rows;  row += blockDim.y * gridDim.y) {\
        out[col * outMajorStride + row] = fun(a, b[col * bMajorStride + row]);\
    }\
  }\
}\




 static __inline__ __device__ double shfl_down(double var, int delta, int width=warpSize)
{
    int hi, lo;
    asm volatile( "mov.b64 { %0, %1 }, %2;" : "=r"(lo), "=r"(hi) : "d"(var) );
    hi = __shfl_down( hi, delta, width );
    lo = __shfl_down( lo, delta, width );
    return __hiloint2double( hi, lo );
}

static __inline__ __device__ int shfl_down(int var, int delta, int width=warpSize)
{
    return __shfl_down(var, delta, width);
}

static __inline__ __device__ unsigned int shfl_down(unsigned int var, int delta, int width=warpSize)
{
    int x = __shfl_down(*(int*)&var, delta, width);
    return *(unsigned int*)(&x);
}

static __inline__ __device__ float shfl_down(float var, int delta, int width=warpSize)
{
    return __shfl_down(var, delta, width);
}

#define laneId (threadIdx.x & 0x1f)



#define REDUCE_FUN(fun, T, identity) \
/* Each column gets 1 block of threads. TODO currently blocksize must be 1 warp*/\
extern "C" \
__global__ void MAKE_NAME(reduce, fun, T) (int rows, int cols,\
    T *out,\
    const T *in, int inMajorStride) {\
  __shared__ T buffer[32];\
\
  T sum = identity;\
  for(int col = threadIdx.y + blockIdx.y * blockDim.y; col < cols; col += blockDim.y * gridDim.y) {\
    for(int row = threadIdx.x + blockIdx.x * blockDim.x; row < rows;  row += blockDim.x * gridDim.x) {\
        sum = fun(sum, in[col * inMajorStride + row]);\
    }\
  }\
  \
  __syncthreads();\
  for (int i = 1; i < blockDim.x; i *= 2) {\
    T x = shfl_down(sum, i);\
    sum = fun(sum, x);\
  }\
  \
  if(laneId == 0) {\
    out[blockIdx.x * gridDim.y + blockIdx.y] = sum;\
  }\
}\
\
/* Each column gets 1 block of threads. TODO currently blocksize must be 1 warp*/\
extern "C" \
__global__ void MAKE_NAME(reduce_col, fun, T) (int rows, int cols,\
    T *out,\
    const T *in, int inMajorStride) {\
  __shared__ T buffer[32];\
\
  for(int col = threadIdx.y + blockIdx.x * blockDim.y; col < cols; col += blockDim.y * gridDim.x) {\
    T sum = identity;\
    for(int row = threadIdx.x; row < rows; row += blockDim.x) {\
      sum = fun(sum, in[col * inMajorStride + row]);\
    }\
    \
    __syncthreads();\
    for (int i = 1; i < blockDim.x; i *= 2) {\
      T x = shfl_down(sum, i);\
      sum = fun(sum, x);\
    }\
    \
    if(laneId == 0) {\
      out[col] = sum;\
    }\
  }\
}\
\
\
/*Each row has its own thread. We should make multiple threads per row, but later. TODO */\
extern "C" \
__global__ void MAKE_NAME(reduce_row, fun, T) (int rows, int cols,\
    T *out,\
    const T *in, int inMajorStride) {\
  __shared__ T buffer[32];\
\
  int numReducers = blockDim.x * gridDim.x;\
  for(int row = threadIdx.x + blockIdx.x * blockDim.x; row < rows; row += numReducers) {\
    T sum = identity;\
    for(int col = 0; col < cols; col++) {\
      sum = fun(sum, in[col * inMajorStride + row]);\
    }\
    \
    out[row] = sum;\
  }\
}\
        

#include "function_decls.cuh"

