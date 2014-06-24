
extern "C" {

static __global__ void enforceLU( double *matrix, int lda )
{
    int i = threadIdx.x;
    int j = blockIdx.x;
    if( i <= j )
        matrix[i + j*lda] = (i == j) ? 1 : 0;
}

}

// zeros out the whole part of matrix above the diagonal (not just a block)
extern "C" {

static __global__ void zerosU(double *matrix, int lda, int elems, int incl)
{
    int dim_x = lda;
    int dim_y = elems / lda;

    int i = blockIdx.x + threadIdx.x;
    int j = blockIdx.y + threadIdx.y;

    if (i >= dim_x || j >= dim_y) return;

    if (i < j)
        matrix[i + j*lda] = 0;
    else if (i == j && incl)
        matrix[i + j*lda] = 0;
}

}

// zeros out the whole part of matrix below the diagonal
extern "C" {

static __global__ void zerosL(double *matrix, int lda, int elems, int incl)
{
    int dim_x = lda;
    int dim_y = elems / lda;

    int i = blockIdx.x + threadIdx.x;
    int j = blockIdx.y + threadIdx.y;

    if (i >= dim_x || j >= dim_y) return;

    if( i > j )
        matrix[i + j*lda] = 0;
    else if (i == j && incl)
        matrix[i + j*lda] = 0;
}

}