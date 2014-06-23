// zeros out the part of a block above the diagonal
// sets ones on the diagonal (kernel by V.Volkov)
extern "C" {

static __global__ void enforceLU( float *matrix, int lda )
{
    int i = threadIdx.x;
    int j = blockIdx.x;
    if( i <= j )
        matrix[i + j*lda] = (i == j) ? 1 : 0;
}

}


// zeros out the whole part of matrix above the diagonal (not just a block)
extern "C" {

static __global__ void zerosU(float *matrix, int lda, int elems, int incl)
{
    int i = blockIdx.x + threadIdx.x;
    int j = blockIdx.y + threadIdx.y;
    if (i + j*lda >= elems) return;

    if (i < j)
        matrix[i + j*lda] = 0;
    else if (i == j && incl)
        matrix[i + j*lda] = 0;
}

}

// zeros out the whole part of matrix below the diagonal
extern "C" {

static __global__ void zerosL(float *matrix, int lda, int elems, int incl)
{
    int i = blockIdx.x + threadIdx.x;
    int j = blockIdx.y + threadIdx.y;
    if (i + j*lda >= elems) return;

    if( i > j )
        matrix[i + j*lda] = 0;
    else if (i == j && incl)
        matrix[i + j*lda] = 0;
}

}