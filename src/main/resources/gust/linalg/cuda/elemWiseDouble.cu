/*
 * Kernel for calulating the element-wise product of two matrices
 * m, n --> dimensions of matrices A, B, C
 */
extern "C" {
__global__ void hadamard(int m, int n, double *A, int lda, double *B, int ldb, double *C, int ldc)
{
    int i = blockIdx.x + threadIdx.x;
    int j = blockIdx.y + threadIdx.y;

    if (i >= m || j >= n) return;

    C[i + j*ldc] = A[i + j*lda] * B[i + j*ldb];
}
}

/*
 * Matrix sum, parameters as above
 */
extern "C" {
 __global__ void matrix_sum(int m, int n, double *A, int lda, double *B, int ldb, double *C, int ldc)
{
    int i = blockIdx.x + threadIdx.x;
    int j = blockIdx.y + threadIdx.y;

    if (i >= m || j >= n) return;

    C[i + j*ldc] = A[i + j*lda] + B[i + j*ldb];
}
}