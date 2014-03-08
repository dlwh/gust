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
}

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




