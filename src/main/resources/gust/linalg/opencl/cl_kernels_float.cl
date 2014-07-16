__kernel void add(__global float* out, __global const float* a, __global const float* b, int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;

    out[idx] = a[idx] + b[idx];
}

__kernel void add_in_place(__global float* a, __global const float* b, int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;

    a[idx] += b[idx];
}

__kernel void sub(__global float* out, __global const float* a, __global const float* b, int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;

    out[idx] = a[idx] - b[idx];
}

__kernel void sub_in_place(__global float* a, __global const float* b, int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;

    a[idx] -= b[idx];
}


/* TODO make this kernel less clumsy */
__kernel void transpose(__global float *output, __global float *input, int r,  int c,  int ldout, int ldin)
{
    int idx = get_global_id(0);
    int row = idx % ldin;
    int col = idx / ldin;

    if (row >= r || col >= c) return;

    output[col + row*ldout] = input[idx];
}