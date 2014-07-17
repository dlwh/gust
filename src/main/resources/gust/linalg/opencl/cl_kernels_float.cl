__kernel void add(__global float *out, int ldout,
                 __global const float *a, int lda,
                 __global const float *b, int ldb,
                 int h, int w, int blockDimX, int blockDimY)
{
    int blockIdx = get_group_id(0);
    int blockIdy = get_group_id(1);

    int threadIdx = get_local_id(0);
    int threadIdy = get_local_id(1);

    int row = blockIdx * blockDimX + threadIdx;
    int col = blockIdy * blockDimY + threadIdy;

    if (row >= h || col >= w) return;

    out[row + col*ldout] = a[row + col*lda] + b[row + col*ldb];
}

__kernel void add_in_place(__global float *out, int ldout,
                         __global const float *in, int ldin,
                         int h, int w, int blockDimX, int blockDimY)
{
    int blockIdx = get_group_id(0);
    int blockIdy = get_group_id(1);

    int threadIdx = get_local_id(0);
    int threadIdy = get_local_id(1);

    int row = blockIdx * blockDimX + threadIdx;
    int col = blockIdy * blockDimY + threadIdy;

    if (row >= h || col >= w) return;

    out[row + col*ldout] += in[row + col*ldin];
}

__kernel void sub(__global float *out, int ldout,
                 __global const float *a, int lda,
                 __global const float *b, int ldb,
                 int h, int w, int blockDimX, int blockDimY)
{
    int blockIdx = get_group_id(0);
    int blockIdy = get_group_id(1);

    int threadIdx = get_local_id(0);
    int threadIdy = get_local_id(1);

    int row = blockIdx * blockDimX + threadIdx;
    int col = blockIdy * blockDimY + threadIdy;

    if (row >= h || col >= w) return;

    out[row + col*ldout] = a[row + col*lda] - b[row + col*ldb];
}

__kernel void sub_in_place(__global float *out, int ldout,
                         __global const float *in, int ldin,
                         int h, int w, int blockDimX, int blockDimY)
{
    int blockIdx = get_group_id(0);
    int blockIdy = get_group_id(1);

    int threadIdx = get_local_id(0);
    int threadIdy = get_local_id(1);

    int row = blockIdx * blockDimX + threadIdx;
    int col = blockIdy * blockDimY + threadIdy;

    if (row >= h || col >= w) return;

    out[row + col*ldout] -= in[row + col*ldin];
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