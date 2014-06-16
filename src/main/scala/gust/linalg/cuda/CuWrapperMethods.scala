package gust.linalg.cuda

import jcuda.jcublas.{cublasOperation, JCublas2, cublasHandle}
import breeze.linalg.DenseMatrix
import jcuda.runtime.{cudaMemcpyKind, JCuda}


/**
 * Created by piotrek on 16.06.2014.
 *
 */
object CuWrapperMethods {
  /*
   * wrapped calls to gemm, don't require passing lda's and elemSizes around
   */
  def SgemmNN(m: Int, n: Int, k: Int, alpha: jcuda.Pointer,
              A: CuMatrix[Float], Aroff: Int, Acoff: Int,
              B: CuMatrix[Float], Broff: Int, Bcoff: Int,
              beta: jcuda.Pointer,
              C: CuMatrix[Float], Croff: Int, Ccoff: Int)(implicit handle: cublasHandle) {

    JCublas2.cublasSgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
      m, n, k, alpha,
      A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff) * A.elemSize), A.majorStride,
      B.offsetPointer.withByteOffset(B.linearIndex(Broff, Bcoff) * B.elemSize), B.majorStride,
      beta,
      C.offsetPointer.withByteOffset(C.linearIndex(Croff, Ccoff) * C.elemSize), C.majorStride)
  }

  def DgemmNN(m: Int, n: Int, k: Int, alpha: jcuda.Pointer,
              A: CuMatrix[Double], Aroff: Int, Acoff: Int,
              B: CuMatrix[Double], Broff: Int, Bcoff: Int,
              beta: jcuda.Pointer,
              C: CuMatrix[Double], Croff: Int, Ccoff: Int)(implicit handle: cublasHandle) {

    JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
      m, n, k, alpha,
      A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff) * A.elemSize), A.majorStride,
      B.offsetPointer.withByteOffset(B.linearIndex(Broff, Bcoff) * B.elemSize), B.majorStride,
      beta,
      C.offsetPointer.withByteOffset(C.linearIndex(Croff, Ccoff) * C.elemSize), C.majorStride)
  }

  def SgemmNT(m: Int, n: Int, k: Int, alpha: jcuda.Pointer,
              A: CuMatrix[Float], Aroff: Int, Acoff: Int,
              B: CuMatrix[Float], Broff: Int, Bcoff: Int,
              beta: jcuda.Pointer,
              C: CuMatrix[Float], Croff: Int, Ccoff: Int)(implicit handle: cublasHandle) {

    JCublas2.cublasSgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T,
      m, n, k, alpha,
      A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff) * A.elemSize), A.majorStride,
      B.offsetPointer.withByteOffset(B.linearIndex(Broff, Bcoff) * B.elemSize), B.majorStride,
      beta,
      C.offsetPointer.withByteOffset(C.linearIndex(Croff, Ccoff) * C.elemSize), C.majorStride)
  }

  def DgemmNT(m: Int, n: Int, k: Int, alpha: jcuda.Pointer,
              A: CuMatrix[Double], Aroff: Int, Acoff: Int,
              B: CuMatrix[Double], Broff: Int, Bcoff: Int,
              beta: jcuda.Pointer,
              C: CuMatrix[Double], Croff: Int, Ccoff: Int)(implicit handle: cublasHandle) {

    JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T,
      m, n, k, alpha,
      A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff) * A.elemSize), A.majorStride,
      B.offsetPointer.withByteOffset(B.linearIndex(Broff, Bcoff) * B.elemSize), B.majorStride,
      beta,
      C.offsetPointer.withByteOffset(C.linearIndex(Croff, Ccoff) * C.elemSize), C.majorStride)
  }

  /*
   * Methods for moving matrices between GPU and CPU
   * Could use get/setMatrix but memcpy works fine
   */
  def uploadFloat(m: Int, n: Int, dst: CuMatrix[Float],
                  dst_roff: Int, dst_coff: Int,
                  src: DenseMatrix[Float], src_roff: Int, src_coff: Int) {
    JCuda.cudaMemcpy2D(dst.offsetPointer.withByteOffset(dst.linearIndex(dst_roff, dst_coff) * dst.elemSize),
      dst.majorStride * dst.elemSize,
      jcuda.Pointer.to(src.data).withByteOffset(src.linearIndex(src_roff, src_coff) * dst.elemSize),
      src.majorStride * dst.elemSize, m * dst.elemSize, n,
      cudaMemcpyKind.cudaMemcpyHostToDevice)
  }

  def uploadDouble(m: Int, n: Int, dst: CuMatrix[Double],
                   dst_roff: Int, dst_coff: Int,
                   src: DenseMatrix[Double], src_roff: Int, src_coff: Int) {
    JCuda.cudaMemcpy2D(dst.offsetPointer.withByteOffset(dst.linearIndex(dst_roff, dst_coff) * dst.elemSize),
      dst.majorStride * dst.elemSize,
      jcuda.Pointer.to(src.data).withByteOffset(src.linearIndex(src_roff, src_coff) * dst.elemSize),
      src.majorStride * dst.elemSize, m * dst.elemSize, n,
      cudaMemcpyKind.cudaMemcpyHostToDevice)
  }

  def downloadFloat(m: Int, n: Int, dst: DenseMatrix[Float], dst_roff: Int, dst_coff: Int,
                    src: CuMatrix[Float], src_roff: Int, src_coff: Int) {
    JCuda.cudaMemcpy2D(jcuda.Pointer.to(dst.data).withByteOffset(dst.linearIndex(dst_roff, dst_coff) * src.elemSize),
      dst.majorStride * src.elemSize,
      src.offsetPointer.withByteOffset(src.linearIndex(src_roff, src_coff) * src.elemSize),
      src.majorStride * src.elemSize, m * src.elemSize, n, cudaMemcpyKind.cudaMemcpyDeviceToHost)
  }

  def downloadDouble(m: Int, n: Int, dst: DenseMatrix[Double], dst_roff: Int, dst_coff: Int,
                     src: CuMatrix[Double], src_roff: Int, src_coff: Int) {
    JCuda.cudaMemcpy2D(jcuda.Pointer.to(dst.data).withByteOffset(dst.linearIndex(dst_roff, dst_coff) * src.elemSize),
      dst.majorStride * src.elemSize,
      src.offsetPointer.withByteOffset(src.linearIndex(src_roff, src_coff) * src.elemSize),
      src.majorStride * src.elemSize, m * src.elemSize, n, cudaMemcpyKind.cudaMemcpyDeviceToHost)
  }

}
