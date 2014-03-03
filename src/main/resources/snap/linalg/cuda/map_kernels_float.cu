#include <stdio.h>

#define MAKE_NAME(fun, T) map_ ## fun ## _ ## T

#define MAP_FUN_1(fun, T) \
extern "C" \
__global__ void MAKE_NAME(fun, T) (int rows, int cols,\
    T *out, int outMajorStride,\
    const T *in, int inMajorStride) {\
    return;\
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











//=== Vector math (one argument) =============================================


// Calculate the arc cosine of the input argument.
extern "C"
__global__ void vec_acosf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = acosf(x[id]);
    }
}


// Calculate the nonnegative arc hyperbolic cosine of the input argument.
extern "C"
__global__ void vec_acoshf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = acoshf(x[id]);
    }
}


// Calculate the arc sine of the input argument.
extern "C"
__global__ void vec_asinf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = asinf(x[id]);
    }
}


// Calculate the arc hyperbolic sine of the input argument.
extern "C"
__global__ void vec_asinhf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = asinhf(x[id]);
    }
}


// Calculate the arc tangent of the input argument.
extern "C"
__global__ void vec_atanf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = atanf(x[id]);
    }
}


// Calculate the arc hyperbolic tangent of the input argument.
extern "C"
__global__ void vec_atanhf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = atanhf(x[id]);
    }
}


// Calculate the cube root of the input argument.
extern "C"
__global__ void vec_cbrtf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = cbrtf(x[id]);
    }
}


// Calculate ceiling of the input argument.
extern "C"
__global__ void vec_ceilf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = ceilf(x[id]);
    }
}


// Calculate the cosine of the input argument.
extern "C"
__global__ void vec_cosf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = cosf(x[id]);
    }
}


// Calculate the hyperbolic cosine of the input argument.
extern "C"
__global__ void vec_coshf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = coshf(x[id]);
    }
}


// Calculate the cosine of the input argument � p .
extern "C"
__global__ void vec_cospif (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = cospif(x[id]);
    }
}


// Calculate the complementary error function of the input argument.
extern "C"
__global__ void vec_erfcf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = erfcf(x[id]);
    }
}


// Calculate the inverse complementary error function of the input argument.
extern "C"
__global__ void vec_erfcinvf (size_t n, TYPE *result, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = erfcinvf(y[id]);
    }
}


// Calculate the scaled complementary error function of the input argument.
extern "C"
__global__ void vec_erfcxf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = erfcxf(x[id]);
    }
}


// Calculate the error function of the input argument.
extern "C"
__global__ void vec_erff (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = erff(x[id]);
    }
}


// Calculate the inverse error function of the input argument.
extern "C"
__global__ void vec_erfinvf (size_t n, TYPE *result, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = erfinvf(y[id]);
    }
}


// Calculate the base 10 exponential of the input argument.
extern "C"
__global__ void vec_exp10f (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = exp10f(x[id]);
    }
}


// Calculate the base 2 exponential of the input argument.
extern "C"
__global__ void vec_exp2f (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = exp2f(x[id]);
    }
}


// Calculate the base e exponential of the input argument.
extern "C"
__global__ void vec_expf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = expf(x[id]);
    }
}


// Calculate the base e exponential of the input argument, minus 1.
extern "C"
__global__ void vec_expm1f (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = expm1f(x[id]);
    }
}


// Calculate the absolute value of its argument.
extern "C"
__global__ void vec_fabsf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = fabsf(x[id]);
    }
}


// Calculate the largest integer less than or equal to x.
extern "C"
__global__ void vec_floorf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = floorf(x[id]);
    }
}


// Calculate the value of the Bessel function of the first kind of order 0 for the input argument.
extern "C"
__global__ void vec_j0f (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = j0f(x[id]);
    }
}


// Calculate the value of the Bessel function of the first kind of order 1 for the input argument.
extern "C"
__global__ void vec_j1f (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = j1f(x[id]);
    }
}


// Calculate the natural logarithm of the absolute value of the gamma function of the input argument.
extern "C"
__global__ void vec_lgammaf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = lgammaf(x[id]);
    }
}


// Calculate the base 10 logarithm of the input argument.
extern "C"
__global__ void vec_log10f (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = log10f(x[id]);
    }
}


// Calculate the value of l o g e ( 1 + x ) .
extern "C"
__global__ void vec_log1pf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = log1pf(x[id]);
    }
}


// Calculate the base 2 logarithm of the input argument.
extern "C"
__global__ void vec_log2f (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = log2f(x[id]);
    }
}


// Calculate the TYPEing point representation of the exponent of the input argument.
extern "C"
__global__ void vec_logbf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = logbf(x[id]);
    }
}


// Calculate the natural logarithm of the input argument.
extern "C"
__global__ void vec_logf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = logf(x[id]);
    }
}


// Calculate the standard normal cumulative distribution function.
extern "C"
__global__ void vec_normcdff (size_t n, TYPE *result, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = normcdff(y[id]);
    }
}


// Calculate the inverse of the standard normal cumulative distribution function.
extern "C"
__global__ void vec_normcdfinvf (size_t n, TYPE *result, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = normcdfinvf(y[id]);
    }
}


// Calculate reciprocal cube root function.
extern "C"
__global__ void vec_rcbrtf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = rcbrtf(x[id]);
    }
}


// Round input to nearest integer value in TYPEing-point.
extern "C"
__global__ void vec_rintf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = rintf(x[id]);
    }
}


// Round to nearest integer value in TYPEing-point.
extern "C"
__global__ void vec_roundf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = roundf(x[id]);
    }
}


// Calculate the reciprocal of the square root of the input argument.
extern "C"
__global__ void vec_rsqrtf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = rsqrtf(x[id]);
    }
}


// Calculate the sine of the input argument.
extern "C"
__global__ void vec_sinf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = sinf(x[id]);
    }
}


// Calculate the hyperbolic sine of the input argument.
extern "C"
__global__ void vec_sinhf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = sinhf(x[id]);
    }
}


// Calculate the sine of the input argument � p .
extern "C"
__global__ void vec_sinpif (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = sinpif(x[id]);
    }
}


// Calculate the square root of the input argument.
extern "C"
__global__ void vec_sqrtf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = sqrtf(x[id]);
    }
}


// Calculate the tangent of the input argument.
extern "C"
__global__ void vec_tanf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = tanf(x[id]);
    }
}


// Calculate the hyperbolic tangent of the input argument.
extern "C"
__global__ void vec_tanhf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = tanhf(x[id]);
    }
}


// Calculate the gamma function of the input argument.
extern "C"
__global__ void vec_tgammaf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = tgammaf(x[id]);
    }
}


// Truncate input argument to the integral part.
extern "C"
__global__ void vec_truncf (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = truncf(x[id]);
    }
}


// Calculate the value of the Bessel function of the second kind of order 0 for the input argument.
extern "C"
__global__ void vec_y0f (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = y0f(x[id]);
    }
}


// Calculate the value of the Bessel function of the second kind of order 1 for the input argument.
extern "C"
__global__ void vec_y1f (size_t n, TYPE *result, TYPE  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = y1f(x[id]);
    }
}











//=== Vector math (two arguments) ============================================





// Create value with given magnitude, copying sign of second value.
extern "C"
__global__ void vec_copysignf ( size_t n, TYPE *result, TYPE  *x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = copysignf(x[id], y[id]);
    }
}

// Compute the positive difference between x and y.
extern "C"
__global__ void vec_fdimf ( size_t n, TYPE *result, TYPE  *x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = fdimf(x[id], y[id]);
    }
}

// Divide two TYPEing point values.
extern "C"
__global__ void vec_fdividef ( size_t n, TYPE *result, TYPE  *x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = fdividef(x[id], y[id]);
    }
}

// Determine the maximum numeric value of the arguments.
extern "C"
__global__ void vec_fmaxf ( size_t n, TYPE *result, TYPE  *x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = fmaxf(x[id], y[id]);
    }
}

// Determine the minimum numeric value of the arguments.
extern "C"
__global__ void vec_fminf ( size_t n, TYPE *result, TYPE  *x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = fminf(x[id], y[id]);
    }
}

// Calculate the TYPEing-point remainder of x / y.
extern "C"
__global__ void vec_fmodf ( size_t n, TYPE *result, TYPE  *x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = fmodf(x[id], y[id]);
    }
}

// Calculate the square root of the sum of squares of two arguments.
extern "C"
__global__ void vec_hypotf ( size_t n, TYPE *result, TYPE  *x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = hypotf(x[id], y[id]);
    }
}

// Return next representable single-precision TYPEing-point value afer argument.
extern "C"
__global__ void vec_nextafterf ( size_t n, TYPE *result, TYPE  *x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = nextafterf(x[id], y[id]);
    }
}

// Calculate the value of first argument to the power of second argument.
extern "C"
__global__ void vec_powf ( size_t n, TYPE *result, TYPE  *x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = powf(x[id], y[id]);
    }
}

// Compute single-precision TYPEing-point remainder.
extern "C"
__global__ void vec_remainderf ( size_t n, TYPE *result, TYPE  *x, TYPE  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = remainderf(x[id], y[id]);
    }
}




