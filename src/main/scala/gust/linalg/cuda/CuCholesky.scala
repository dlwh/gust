package gust.linalg.cuda

import breeze.linalg.{NotConvergedException, DenseMatrix}
import jcuda.jcublas._
import jcuda.jcusparse._
import org.netlib.util.intW
import spire.syntax.cfor._
import gust.linalg.cuda.CuWrapperMethods._
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}

import scala.reflect.ClassTag

/**
 * Created by Piotr on 2014-07-16.
 */
object CuCholesky {

  /**
   * Returns a Cholesky decomposition of matrix A,
   * i.e. returns L such that A = L * L'
   *
   * Based on V. Volkov's 'gpu_spotrf' code.
   *
   * @param A symmetric, positive definite matrix
   * @return lower triangular matrix
   */
  def choleskyFloat(A: CuMatrix[Float])(implicit handle: cublasHandle): CuMatrix[Float] = {
    if (A.rows != A.cols) {
      println("A has to be a square matrix")
      return A
    }

    // Don't know if we want to check whether the matrix is
    // symmetric AND positive definite -- symmetry is easy to check
    // but positive definiteness is not, so I just test if we have a reasonably correct factorization
    if (!A.isSymmetric) {
      println("A has to be a symmetric matrix")
      return A
    }

    val n = A.rows
    val nb = if (n < 64) n else 64  // adjust this

    val oneArr = Array(1.0f)
    val one = jcuda.Pointer.to(oneArr)
    val zeroArr = Array(0.0f)
    val zero = jcuda.Pointer.to(zeroArr)
    val minusOneArr = Array(-1.0f)
    val minusOne = jcuda.Pointer.to(minusOneArr)

    val cpu_mat = A.toDense
    val gpu_mat = CuMatrix.create[Float](n, n)

    val temp = CuMatrix.zeros[Float](n, n)

    val info = new intW(0)

    uploadLFloat(n-nb, gpu_mat, nb, nb, cpu_mat, nb, nb)

    cfor(0)(_ < n, _ + nb) { i => {
      val h = n - i
      val w = if (h < nb) h else nb

      if (i > 0) {
        JCublas2.cublasSsyrk(handle,cublasFillMode.CUBLAS_FILL_MODE_LOWER, cublasOperation.CUBLAS_OP_N,
          w, nb, minusOne,
          gpu_mat.offsetPointer.withByteOffset(gpu_mat.linearIndex(i, i-nb) * gpu_mat.elemSize), gpu_mat.majorStride,
          one,
          gpu_mat.offsetPointer.withByteOffset(gpu_mat.linearIndex(i, i) * gpu_mat.elemSize), gpu_mat.majorStride)

        SgemmNT(h-w, w, nb, minusOne, gpu_mat, i+nb, i-nb, gpu_mat, i, i-nb, one, gpu_mat, i+nb, i)

        downloadFloat(h, w, cpu_mat, i, i, gpu_mat, i, i)

        if(h > nb)
          JCublas2.cublasSsyrk(handle,cublasFillMode.CUBLAS_FILL_MODE_LOWER, cublasOperation.CUBLAS_OP_N,
            h-nb, nb, minusOne,
            gpu_mat.offsetPointer.withByteOffset(gpu_mat.linearIndex(i+nb, i-nb) * gpu_mat.elemSize), gpu_mat.majorStride,
            one,
            gpu_mat.offsetPointer.withByteOffset(gpu_mat.linearIndex(i+nb, i+nb) * gpu_mat.elemSize), gpu_mat.majorStride)

      }

      lapack.spotrf("L", w, cpu_mat.data, i + i*cpu_mat.majorStride, cpu_mat.majorStride, info)

      if( h > nb )
      {
        // the main problem here is that apparently the trsm function is missing
        // from this lapack
        val m = h - nb
        // upload a part of the matrix to a temp matrix:
        uploadFloat(h, w, temp, i, i, cpu_mat, i, i)

        JCublas2.cublasStrsm(handle, cublasSideMode.CUBLAS_SIDE_RIGHT, cublasFillMode.CUBLAS_FILL_MODE_LOWER, cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N,
          m, nb, one,
          temp.offsetPointer.withByteOffset(temp.linearIndex(i, i) * temp.elemSize), temp.majorStride,
          temp.offsetPointer.withByteOffset(temp.linearIndex(i+nb,i) * temp.elemSize), temp.majorStride)

        downloadFloat(h, w, cpu_mat, i, i, temp, i, i)
        copyFloat(h, w, gpu_mat, i, i, temp, i, i)

      }
    }}

    val d_L = CuMatrix.fromDense(cpu_mat)
    zeroOutFloat(d_L, 'U')

    /*
     * TODO perhaps it would be better to normalize the residuals using A's norm or something like that
     * this threshold is a first guess and may not make any sense at all
     */
    if (residualFloat(A, d_L, d_L.t) > 1e-7)
      throw new NotConvergedException(NotConvergedException.Divergence, "Matrix is not positive-definite")

    d_L
  }

  def choleskyDouble(A: CuMatrix[Double])(implicit handle: cublasHandle): CuMatrix[Double] = {
    if (A.rows != A.cols) {
      println("A has to be a square matrix")
      return A
    }

    if (!A.isSymmetric) {
      println("A has to be a symmetric matrix")
      return A
    }

    val n = A.rows
    val nb = if (n < 64) n else 64  // adjust this

    val oneArr = Array(1.0)
    val one = jcuda.Pointer.to(oneArr)
    val zeroArr = Array(0.0)
    val zero = jcuda.Pointer.to(zeroArr)
    val minusOneArr = Array(-1.0)
    val minusOne = jcuda.Pointer.to(minusOneArr)

    val cpu_mat = A.toDense
    val gpu_mat = CuMatrix.create[Double](n, n)

    val temp = CuMatrix.zeros[Double](n, n)

    val info = new intW(0)

    uploadLDouble(n-nb, gpu_mat, nb, nb, cpu_mat, nb, nb)

    cfor(0)(_ < n, _ + nb) { i => {
      val h = n - i
      val w = if (h < nb) h else nb

      if (i > 0) {
        JCublas2.cublasDsyrk(handle,cublasFillMode.CUBLAS_FILL_MODE_LOWER, cublasOperation.CUBLAS_OP_N,
          w, nb, minusOne,
          gpu_mat.offsetPointer.withByteOffset(gpu_mat.linearIndex(i, i-nb) * gpu_mat.elemSize), gpu_mat.majorStride,
          one,
          gpu_mat.offsetPointer.withByteOffset(gpu_mat.linearIndex(i, i) * gpu_mat.elemSize), gpu_mat.majorStride)

        DgemmNT(h-w, w, nb, minusOne, gpu_mat, i+nb, i-nb, gpu_mat, i, i-nb, one, gpu_mat, i+nb, i)

        downloadDouble(h, w, cpu_mat, i, i, gpu_mat, i, i)

        if(h > nb)
          JCublas2.cublasDsyrk(handle,cublasFillMode.CUBLAS_FILL_MODE_LOWER, cublasOperation.CUBLAS_OP_N,
            h-nb, nb, minusOne,
            gpu_mat.offsetPointer.withByteOffset(gpu_mat.linearIndex(i+nb, i-nb) * gpu_mat.elemSize), gpu_mat.majorStride,
            one,
            gpu_mat.offsetPointer.withByteOffset(gpu_mat.linearIndex(i+nb, i+nb) * gpu_mat.elemSize), gpu_mat.majorStride)

      }

      lapack.dpotrf("L", w, cpu_mat.data, i + i*cpu_mat.majorStride, cpu_mat.majorStride, info)

      if( h > nb )
      {
        val m = h - nb

        uploadDouble(h, w, temp, i, i, cpu_mat, i, i)

        JCublas2.cublasDtrsm(handle, cublasSideMode.CUBLAS_SIDE_RIGHT, cublasFillMode.CUBLAS_FILL_MODE_LOWER, cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N,
          m, nb, one,
          temp.offsetPointer.withByteOffset(temp.linearIndex(i, i) * temp.elemSize), temp.majorStride,
          temp.offsetPointer.withByteOffset(temp.linearIndex(i+nb,i) * temp.elemSize), temp.majorStride)

        downloadDouble(h, w, cpu_mat, i, i, temp, i, i)
        copyDouble(h, w, gpu_mat, i, i, temp, i, i)

      }
    }}

    val d_L = CuMatrix.fromDense(cpu_mat)
    zeroOutDouble(d_L, 'U')

    if (residualDouble(A, d_L, d_L.t) > 1e-7)
      throw new NotConvergedException(NotConvergedException.Divergence, "Matrix is not positive-definite")

    d_L
  }

  def incompleteCholeskyFloat(A: CuSparseMatrix[Float])(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle): CuSparseMatrix[Float] = {
    if (A.rows != A.cols) {
      println("A has to be a square matrix")
      return A
    }

    val AS = A.copy

    // for now the matrix will be treated as symmetric even if it's not
    JCusparse2.cusparseSetMatType(AS.descr, cusparseMatrixType.CUSPARSE_MATRIX_TYPE_SYMMETRIC)
    JCusparse2.cusparseSetMatFillMode(AS.descr, cusparseFillMode.CUSPARSE_FILL_MODE_UPPER)

    val info = new cusparseSolveAnalysisInfo
    JCusparse2.cusparseCreateSolveAnalysisInfo(info)
    val trans = if (AS.isTranspose) cusparseOperation.CUSPARSE_OPERATION_TRANSPOSE else cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE
    val m = AS.rows
    JCusparse2.cusparseScsrsv_analysis(sparseHandle, trans, m, AS.nnz, AS.descr, AS.cscVal.offsetPointer, AS.cscColPtr.offsetPointer, AS.cscRowInd.offsetPointer, info)
    JCusparse2.cusparseScsric0(sparseHandle, trans, m, AS.descr, AS.cscVal.offsetPointer, AS.cscColPtr.offsetPointer, AS.cscRowInd.offsetPointer, info)

    JCusparse2.cusparseDestroySolveAnalysisInfo(info)

    AS
  }

  def incompleteCholeskyDouble(A: CuSparseMatrix[Double])(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle): CuSparseMatrix[Double] = {
    if (A.rows != A.cols) {
      println("A has to be a square matrix")
      return A
    }

    val AS = A.copy

    // for now the matrix will be treated as symmetric even if it's not
    JCusparse2.cusparseSetMatType(AS.descr, cusparseMatrixType.CUSPARSE_MATRIX_TYPE_SYMMETRIC)
    JCusparse2.cusparseSetMatFillMode(AS.descr, cusparseFillMode.CUSPARSE_FILL_MODE_UPPER)

    val info = new cusparseSolveAnalysisInfo
    JCusparse2.cusparseCreateSolveAnalysisInfo(info)
    val trans = if (AS.isTranspose) cusparseOperation.CUSPARSE_OPERATION_TRANSPOSE else cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE
    val m = AS.rows
    JCusparse2.cusparseDcsrsv_analysis(sparseHandle, trans, m, AS.nnz, AS.descr, AS.cscVal.offsetPointer, AS.cscColPtr.offsetPointer, AS.cscRowInd.offsetPointer, info)
    JCusparse2.cusparseDcsric0(sparseHandle, trans, m, AS.descr, AS.cscVal.offsetPointer, AS.cscColPtr.offsetPointer, AS.cscRowInd.offsetPointer, info)

    JCusparse2.cusparseDestroySolveAnalysisInfo(info)

    AS
  }


  private def uploadLFloat(n: Int, dst: CuMatrix[Float], dstRoff: Int, dstCoff: Int, src: DenseMatrix[Float], srcRoff: Int, srcCoff: Int) {
    val nb = 128
    cfor (0)(_ < n, _ + nb) { i => {
      uploadFloat(n - i, if (nb < n - i) nb else n - i, dst, i+dstRoff, i+dstCoff, src, i+srcRoff, i+srcCoff)
    }}
  }

  private def uploadLDouble(n: Int, dst: CuMatrix[Double], dstRoff: Int, dstCoff: Int, src: DenseMatrix[Double], srcRoff: Int, srcCoff: Int) {
    val nb = 128
    cfor (0)(_ < n, _ + nb) { i => {
      uploadDouble(n - i, if (nb < n - i) nb else n - i, dst, i+dstRoff, i+dstCoff, src, i+srcRoff, i+srcCoff)
    }}
  }
}
