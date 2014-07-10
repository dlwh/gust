package gust.linalg.cuda

import breeze.generic.UFunc
import jcuda.jcublas.{JCublas2, cublasHandle, cublasOperation}
import breeze.linalg.{DenseMatrix, DenseVector}
import jcuda.runtime.{cudaMemcpyKind, JCuda}
import org.netlib.util.intW
import jcuda.driver.{CUfunction, CUmodule, JCudaDriver}
import gust.util.cuda.{CuContext, CuDevice}
import spire.syntax.cfor._
import gust.linalg.cuda.CuWrapperMethods._
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}

/**
 * Created by Piotr on 2014-07-07.
 */
object CuSVD {

  /**
   * Computes the bidiagonalization of matrix A such that:
   * B = P*A*Q
   * @param A
   * @param handle
   * @return
   */
  def bidiagFloat(A: CuMatrix[Float])(implicit handle: cublasHandle): (CuMatrix[Float], CuMatrix[Float], CuMatrix[Float]) = {
    if (A.rows < A.cols) {
      println("Number of rows of matrix A cannot be smaller than the number of columns.")
      return (null, A, null)
    }

    val m = A.rows
    val n = A.cols

    val d_A = CuMatrix.create[Float](m, n); d_A := A
    val d_Q = CuMatrix.create[Float](m, m); eyeizeFloat(d_Q)
    val d_P = CuMatrix.create[Float](n, n); eyeizeFloat(d_P)

    val oneArr = Array(1.0f)
    val one = jcuda.Pointer.to(oneArr)
    val zeroArr = Array(0.0f)
    val zero = jcuda.Pointer.to(zeroArr)
    val minusOneArr = Array(-1.0f)
    val minusOne = jcuda.Pointer.to(minusOneArr)

    val d_v = CuMatrix.create[Float](m, 1)  // here we will store the householder vectors
    val eyeM = CuMatrix.create[Float](m, m)
    val eyeN = CuMatrix.create[Float](n, n) // having two will be good for tall-skinny matrices
    val d_Q1 = CuMatrix.create[Float](m, m)
    val d_P1 = CuMatrix.create[Float](n, n)
    val betaArr = Array(0.0f)

    cfor(0)(_ < n, _ + 1) { i => {
      // eliminate a column:
      householderMatFloat(d_A, i, i, d_v, d_Q1)

      //d_A = d_Q1 * d_A
      //SgemmNN(m, n, m, one, d_Q1, 0, 0, d_A, 0, 0, zero, d_A, 0, 0)
      SgemmNN(m-i, n-i, m-i, one, d_Q1, i, i, d_A, i, i, zero, d_A, i, i)
      // d_Q = d_Q1 * d_Q
      SgemmNN(m, m, m, one, d_Q1, 0, 0, d_Q, 0, 0, zero, d_Q, 0, 0)
      //SgemmNN(m-i, m-i, m-i, one, d_Q1, i, i, d_Q, i, i, zero, d_Q, i, i)

      if (i < n - 2) {
        // eliminate a row:
        householderMatFloat(d_A, i, i + 1, d_v, d_P1, col = false)

        // d_A = d_A * d_P1
        //SgemmNN(m, n, n, one, d_A, 0, 0, d_P1, 0, 0, zero, d_A, 0, 0)
        SgemmNN(m-i, n-i-1, n-i-1, one, d_A, i, i+1, d_P1, i+1, i+1, zero, d_A, i, i+1)
        // d_P = d_P * d_P1
        SgemmNN(n, n, n, one, d_P, 0, 0, d_P1, 0, 0, zero, d_P, 0, 0)
        //SgemmNN(n-i-1, n -i-1, n-i-1, one, d_P, i+1, i+1, d_P1, i+1, i+1, zero, d_P, i+1, i+1)
      }
    }}

    // make d_A explicitly bidiagonal (there're some close-to-zero values due to roundoff errors)
    zeroOutFloatOffset(d_A, 0, 1, 'U')
    zeroOutFloat(d_A, 'L')

    (d_Q, d_A, d_P)
  }


  /**
   * Generates a householder matrix from the given row or column of the matrix A
   * @param A
   * @param Aroff
   * @param Acoff
   * @param d_v This is just a workspace but we don't want to reallocate it every time
   * @param d_Q1 Here we will return the householder matrix. Once again --> to avoid reallocating
   * @param col if true, we take the vector from a column of A. Otherwise, from a row
   */
  def householderMatFloat(A: CuMatrix[Float], Aroff: Int, Acoff: Int, d_v: CuMatrix[Float],
                             d_Q1: CuMatrix[Float],  col: Boolean = true)(implicit handle: cublasHandle): Unit = {
    if (Aroff >= A.rows || Acoff >= A.cols) {
      println("Offset out of bounds")
      return
    }

    val m = A.rows
    val n = A.cols

    val len = if (col) m else n
    val stride = if (col) 1 else A.majorStride
    val offset = if (col) Aroff else Acoff
    val oneArr = Array(1.0f)
    val minusTwoArr = Array(-2.0f)
    //val d_v = CuMatrix.create[Float](len, 1)

    // copy the vector from the matrix into d_v:
    JCublas2.cublasScopy(handle, len-offset, A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff) * A.elemSize),
                         stride, d_v.offsetPointer.withByteOffset(d_v.linearIndex(offset, 0) * d_v.elemSize), 1)

    // normArr(0) = |d_v|
    val normArr = Array(0.0f)
    JCublas2.cublasSnrm2(handle, len-offset, d_v.offsetPointer.withByteOffset(d_v.linearIndex(offset, 0) * d_v.elemSize),
                         1, jcuda.Pointer.to(normArr))

    // v0Arr(0) = d_v(0)
    val v0Arr = Array(0.0f)
    JCuda.cudaMemcpy2D(jcuda.Pointer.to(v0Arr), d_v.majorStride * d_v.elemSize,
      d_v.offsetPointer.withByteOffset(d_v.linearIndex(offset, 0) * d_v.elemSize),
      d_v.majorStride * d_v.elemSize, d_v.elemSize, 1, cudaMemcpyKind.cudaMemcpyDeviceToHost)

    v0Arr(0) -= normArr(0)
    // put d_v(0) back in the vector
    JCuda.cudaMemcpy2D(d_v.offsetPointer.withByteOffset(d_v.linearIndex(offset, 0) * d_v.elemSize),
      d_v.majorStride * d_v.elemSize, jcuda.Pointer.to(v0Arr),
      d_v.majorStride * d_v.elemSize, d_v.elemSize, 1, cudaMemcpyKind.cudaMemcpyHostToDevice)

    // calculate the norm of the updated vector v:
    JCublas2.cublasSnrm2(handle, len-offset, d_v.offsetPointer.withByteOffset(d_v.linearIndex(offset, 0) * d_v.elemSize),
      1, jcuda.Pointer.to(normArr))

    normArr(0) = 1.0f / normArr(0)

    // d_v = d_v / |d_v|
    JCublas2.cublasSscal(handle, len-offset, jcuda.Pointer.to(normArr),
      d_v.offsetPointer.withByteOffset(d_v.linearIndex(offset, 0) * d_v.elemSize), 1)

    // make sure d_Q1 is an identity matrix:
    eyeizeFloat(d_Q1)
    // d_Q1 = I - 2*d_v*d_v' (d_Q == I)
    SgemmNT(len-offset, len-offset, 1, jcuda.Pointer.to(minusTwoArr), d_v, offset, 0, d_v, offset, 0,
            jcuda.Pointer.to(oneArr), d_Q1, offset, offset)

  }
}
