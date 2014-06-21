
extern "C" {

static __global__ void enforceLU( float *matrix, int lda )
{
    int i = threadIdx.x;
    int j = blockIdx.x;
    if( i <= j )
        matrix[i + j*lda] = (i == j) ? 1 : 0;
}

}
