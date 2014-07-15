__kernel void transpose(__global float *output, __global float *input, int r,  int c,  int ldout, int ldin)
{
    int idx = get_global_id(0);
    int row = idx % ldin;
    int col = idx / ldin;

    if (row >= r || col >= c) return;

    output[col + row*ldout] = input[idx];
}