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

MAP_FUN_1(acos, TYPE)
MAP_FUN_1(acosh, TYPE)
MAP_FUN_1(asin, TYPE)
MAP_FUN_1(asinh, TYPE)
MAP_FUN_1(atan, TYPE)
MAP_FUN_1(atanh, TYPE)
MAP_FUN_1(cbrt, TYPE)
MAP_FUN_1(ceil, TYPE)
MAP_FUN_1(cos, TYPE)
MAP_FUN_1(cosh, TYPE)
MAP_FUN_1(cospi, TYPE)
MAP_FUN_1(erfc, TYPE)
MAP_FUN_1(erfcinv, TYPE)
MAP_FUN_1(erfcx, TYPE)
MAP_FUN_1(erf, TYPE)
MAP_FUN_1(erfinv, TYPE)
MAP_FUN_1(exp10, TYPE)
MAP_FUN_1(exp2, TYPE)
MAP_FUN_1(exp, TYPE)
MAP_FUN_1(expm1, TYPE)
MAP_FUN_1(fabs, TYPE)
MAP_FUN_1(floor, TYPE)
MAP_FUN_1(j0, TYPE)
MAP_FUN_1(j1, TYPE)
MAP_FUN_1(lgamma, TYPE)
MAP_FUN_1(log10, TYPE)
MAP_FUN_1(log1p, TYPE)
MAP_FUN_1(log2, TYPE)
MAP_FUN_1(logb, TYPE)
MAP_FUN_1(log, TYPE)
MAP_FUN_1(nearbyint, TYPE)
MAP_FUN_1(normcdf, TYPE)
MAP_FUN_1(normcdfinv, TYPE)
MAP_FUN_1(rcbrt, TYPE)
MAP_FUN_1(rint, TYPE)
MAP_FUN_1(round, TYPE)
MAP_FUN_1(rsqrt, TYPE)
MAP_FUN_1(sin, TYPE)
MAP_FUN_1(sinh, TYPE)
MAP_FUN_1(sinpi, TYPE)
MAP_FUN_1(sqrt, TYPE)
MAP_FUN_1(tan, TYPE)
MAP_FUN_1(tanh, TYPE)
MAP_FUN_1(tgamma, TYPE)
MAP_FUN_1(trunc, TYPE)
MAP_FUN_1(y0, TYPE)
MAP_FUN_1(y1, TYPE)



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


__device__ inline TYPE add(TYPE a, TYPE b) { return a + b; }
__device__ inline TYPE sub(TYPE a, TYPE b) { return a - b; }
__device__ inline TYPE mul(TYPE a, TYPE b) { return a * b; }
__device__ inline TYPE div(TYPE a, TYPE b) { return a / b; }
__device__ inline TYPE mod(TYPE a, TYPE b) { return fmod(a, b); }

MAP_FUN_2(add, TYPE)
MAP_FUN_2(sub, TYPE)
MAP_FUN_2(mul, TYPE)
MAP_FUN_2(div, TYPE)
MAP_FUN_2(mod, TYPE)
MAP_FUN_2(pow, TYPE)
MAP_FUN_2(max, TYPE)
MAP_FUN_2(min, TYPE)

 static __inline__ __device__ double shfl_down(double var, int delta, int width=warpSize)
{
    printf("aaa\n");
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

REDUCE_FUN(add, TYPE, 0)         
REDUCE_FUN(max, TYPE, -INFINITY)         
REDUCE_FUN(min, TYPE, INFINITY)         


// TODO: add back in set

//=== Vector arithmetic ======================================================

extern "C"
__global__ void vec_addf (size_t n, TYPE *result, TYPE  *x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] + y[id];
    }
}


extern "C"
__global__ void vec_subf (size_t n, TYPE *result, TYPE  *x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] - y[id];
    }
}


extern "C"
__global__ void vec_mulf (size_t n, TYPE *result, TYPE  *x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] * y[id];
    }
}


extern "C"
__global__ void vec_divf (size_t n, TYPE *result, TYPE  *x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] / y[id];
    }
}

extern "C"
__global__ void vec_negatef (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = -x[id];
    }
}




//=== Vector-and-scalar arithmetic ===========================================

extern "C"
__global__ void vec_addScalarf (size_t n, TYPE *result, TYPE  *x, TYPE  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] + y;
    }
}


extern "C"
__global__ void vec_subScalarf (size_t n, TYPE *result, TYPE  *x, TYPE  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] - y;
    }
}


extern "C"
__global__ void vec_mulScalarf (size_t n, TYPE *result, TYPE  *x, TYPE  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] * y;
    }
}


extern "C"
__global__ void vec_divScalarf (size_t n, TYPE *result, TYPE  *x, TYPE  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] / y;
    }
}




extern "C"
__global__ void vec_scalarAddf (size_t n, TYPE *result, TYPE  x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x + y[id];
    }
}


extern "C"
__global__ void vec_scalarSubf (size_t n, TYPE *result, TYPE  x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x - y[id];
    }
}


extern "C"
__global__ void vec_scalarMulf (size_t n, TYPE *result, TYPE  x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x * y[id];
    }
}


extern "C"
__global__ void vec_scalarDivf (size_t n, TYPE *result, TYPE  x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x / y[id];
    }
}











//=== Vector comparison ======================================================

extern "C"
__global__ void vec_ltf (size_t n, TYPE *result, TYPE  *x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] < y[id])?1.0f:0.0f;
    }
}


extern "C"
__global__ void vec_ltef (size_t n, TYPE *result, TYPE  *x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] <= y[id])?1.0f:0.0f;
    }
}


extern "C"
__global__ void vec_eqf (size_t n, TYPE *result, TYPE  *x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] == y[id])?1.0f:0.0f;
    }
}


extern "C"
__global__ void vec_gtef (size_t n, TYPE *result, TYPE  *x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] >= y[id])?1.0f:0.0f;
    }
}


extern "C"
__global__ void vec_gtf (size_t n, TYPE *result, TYPE  *x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] > y[id])?1.0f:0.0f;
    }
}



extern "C"
__global__ void vec_nef (size_t n, TYPE *result, TYPE  *x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] != y[id])?1.0f:0.0f;
    }
}




//=== Vector-and-scalar comparison ===========================================

extern "C"
__global__ void vec_ltScalarf (size_t n, TYPE *result, TYPE  *x, TYPE  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] < y)?1.0f:0.0f;
    }
}


extern "C"
__global__ void vec_lteScalarf (size_t n, TYPE *result, TYPE  *x, TYPE  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] <= y)?1.0f:0.0f;
    }
}


extern "C"
__global__ void vec_eqScalarf (size_t n, TYPE *result, TYPE  *x, TYPE  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] == y)?1.0f:0.0f;
    }
}


extern "C"
__global__ void vec_gteScalarf (size_t n, TYPE *result, TYPE  *x, TYPE  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] >= y)?1.0f:0.0f;
    }
}


extern "C"
__global__ void vec_gtScalarf (size_t n, TYPE *result, TYPE  *x, TYPE  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] > y)?1.0f:0.0f;
    }
}


extern "C"
__global__ void vec_neScalarf (size_t n, TYPE *result, TYPE  *x, TYPE  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] != y)?1.0f:0.0f;
    }
}




