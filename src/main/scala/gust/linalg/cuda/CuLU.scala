package gust.linalg.cuda

import breeze.linalg.support.{CanCollapseAxis, CanTranspose, CanSlice2}


import jcuda.jcublas.{cublasOperation, cublasHandle, JCublas2, cublasDiagType, cublasFillMode, cublasSideMode}
import gust.util.cuda
import jcuda.runtime.{cudaError, cudaMemcpyKind, cudaStream_t, JCuda}
import cuda._
import breeze.linalg.operators._
import breeze.linalg._
import breeze.linalg.support.{CanCollapseAxis, CanTranspose, CanSlice2}
import org.bridj.Pointer

import jcuda.driver._
import jcuda.jcurand.{curandRngType, curandGenerator}
import breeze.math.{Semiring, Ring}
import breeze.numerics._
import breeze.generic.UFunc
import scala.reflect._

import com.github.fommil.netlib.LAPACK.{getInstance=>lapack}
import com.github.fommil.netlib.ARPACK
import org.netlib.util.intW
import spire.syntax.cfor._
import CuWrapperMethods._
import breeze.linalg._

/**
 * Created by piotrek on 21.05.2014.
 */
object CuLU extends UFunc {


  /**
   * Constructs full L and U factors from one matrix returned by our methods
   */
  def LUFactorsDouble(A: CuMatrix[Double])(implicit handle: cublasHandle): (CuMatrix[Double], CuMatrix[Double]) = {
    val d_L = CuMatrix.create[Double](A.rows, A.cols); d_L := A
    val d_U = CuMatrix.create[Double](A.rows, A.cols); d_U := A

    // zero out appropriate parts of matrices
    CuWrapperMethods.zeroOutDouble(d_L, 'U')
    CuWrapperMethods.zeroOutDouble(d_U, 'L')

    // set the diagonal in d_L to ones
    val diag_len = if (A.rows < A.cols) A.rows else A.cols
    val d_diag = DenseMatrix.ones[Double](diag_len, 1)
    JCublas2.cublasSetVector(diag_len, d_L.elemSize.toInt, jcuda.Pointer.to(d_diag.data), 1, d_L.offsetPointer, d_L.majorStride + 1)

    (d_L, d_U)
  }

  def LUFactorsFloat(A: CuMatrix[Float])(implicit handle: cublasHandle): (CuMatrix[Float], CuMatrix[Float]) = {
    val d_L = CuMatrix.create[Float](A.rows, A.cols); d_L := A
    val d_U = CuMatrix.create[Float](A.rows, A.cols); d_U := A

    // zero out appropriate parts of matrices
    CuWrapperMethods.zeroOutFloat(d_L, 'U')
    CuWrapperMethods.zeroOutFloat(d_U, 'L')

    // set the diagonal in d_L to ones
    val diag_len = if (A.rows < A.cols) A.rows else A.cols
    val d_diag = DenseMatrix.ones[Float](diag_len, 1)
    JCublas2.cublasSetVector(diag_len, d_L.elemSize.toInt, jcuda.Pointer.to(d_diag.data), 1, d_L.offsetPointer, d_L.majorStride + 1)

    (d_L, d_U)
  }

  /**
   * LU factorization, based on algorithm from noctua-blog.com
   * One important thing is that the matrix doesn't have to be square.
   * This is algorithm uses pivoting and block-approach.
   *
   * This is actually almost the same as Volkov's LU with the difference of the
   * decomposition of the block on the diagonal being performed GPU-side (which is fine, since we
   * don't use the super-fast mkl that he uses).
   *
   * The returned matrix contains both L and U -- diagonal of L is implicit, as in this
   * variant it's all ones.
   */
  def LUFloat(A: CuMatrix[Float])(implicit handle: cublasHandle): (CuMatrix[Float], CuMatrix[Float]) = {
    val P: Array[Int] = (0 until A.rows).toArray
    val d_A = CuMatrix.create[Float](A.rows, A.cols)
    d_A := A
    val blockSize = 64
    val es = d_A.elemSize
    val lda = A.rows

    val one = Array(1.0f)
    val onePtr = jcuda.Pointer.to(one)
    val minusOne = Array(-1.0f)
    val minusOnePtr = jcuda.Pointer.to(minusOne)

    // this one should be in place to inject the results into the big matrix
    // passing the pointer instead of the whole matrix allows us to
    // do C-style pointer manipulation
    def LUSingleBlockFloat(M: Int, N: Int, ADataPointer: CuPointer, pOffset: Int): Unit = {
      //      val d_A = CuMatrix.create[Float](A.rows, A.cols)
      //      // creating a copy of the matrix
      //      //    JCuda.cudaMemcpy2D(d_A.data.toCuPointer, A.majorStride * A.elemSize,
      //      //      A.data.toCuPointer,
      //      //      A.majorStride * A.elemSize, A.cols * A.elemSize, A.rows, cudaMemcpyKind.cudaMemcpyDeviceToDevice)
      //      d_A := A

      //      val M = A.rows
      //      val N = A.cols
      val minDim = if (M < N) M else N
      //val ADataPointer = d_A.offsetPointer
      val intRes = Array(0)
      val intResPtr = jcuda.Pointer.to(intRes)
      val A_ii = Array(0.0f)
      val AiiPtr = jcuda.Pointer.to(A_ii)
      val alpha = Array(0.0f)

      cfor(0)(_ < minDim, _ + 1) { i => {
        JCublas2.cublasIsamax(handle, M - i, ADataPointer.withByteOffset(d_A.linearIndex(i, i) * es),
          1, intResPtr)

        val pivotRow = i + intRes(0) - 1 // -1 because of cublas' 1-based indexing
        val ip1 = i + 1
        P(i + pOffset) = pivotRow + pOffset

        if (pivotRow != i) {
          // If you put N instead of A.cols, you have to uncomment the lines in LUBlocked
          JCublas2.cublasSswap(handle, N, ADataPointer.withByteOffset(d_A.linearIndex(pivotRow, 0) * es),
            lda, ADataPointer.withByteOffset(d_A.linearIndex(i, 0) * es), lda)
        }

        JCuda.cudaMemcpy2D(AiiPtr, d_A.majorStride * d_A.elemSize,
          ADataPointer.withByteOffset(d_A.linearIndex(i, i) * es),
          d_A.majorStride * es, d_A.elemSize, 1, cudaMemcpyKind.cudaMemcpyDeviceToHost)
        // this one works as well:
        //      curesult =  JCublas2.cublasGetVector(1, es.toInt, ADataPointer.withByteOffset(d_A.linearIndex(i, i) * es),
        //                               1, AiiPtr, 1)

        if (Math.abs(A_ii(0)) < 1e-20) {
          println("Matrix is singular")
          return
        }

        if (ip1 < M) {
          alpha(0) = 1.0f / A_ii(0)
          JCublas2.cublasSscal(handle, M - ip1, jcuda.Pointer.to(alpha),
            ADataPointer.withByteOffset(d_A.linearIndex(ip1, i) * es), 1)
        }

        if (ip1 < minDim) {
          JCublas2.cublasSger(handle, M - ip1, N - ip1, minusOnePtr,
            ADataPointer.withByteOffset(d_A.linearIndex(ip1, i) * es), 1,
            ADataPointer.withByteOffset(d_A.linearIndex(i, ip1) * es), lda,
            ADataPointer.withByteOffset(d_A.linearIndex(ip1, ip1) * es), lda)

        }
      }
      }

      //d_A
    }

    def LUBlockedFloat(d_A: CuMatrix[Float]): Unit = {
      val M = d_A.rows
      val N = d_A.cols
      val minSize = if (M < N) M else N
      val ADataPointer = d_A.offsetPointer

      if (blockSize >= minSize) {
        LUSingleBlockFloat(M, N, ADataPointer, 0)
        return
      }

      cfor(0)(_ < minSize, _ + blockSize) { i => {
        val realBlockSize = if (minSize - i < blockSize) minSize - i else blockSize
        LUSingleBlockFloat(M - i, realBlockSize, ADataPointer.withByteOffset(d_A.linearIndex(i, i) * es), i)

        // uncomment these lines if necessary
        val mm = if (M < i + realBlockSize) M else i + realBlockSize

        cfor(i)(_ < mm - 1, _ + 1) { p => {
          // P(p) = P(p) + i // ???
          if (P(p) != p) {
            JCublas2.cublasSswap(handle, i, ADataPointer.withByteOffset(d_A.linearIndex(p, 0) * es),
              lda, ADataPointer.withByteOffset(d_A.linearIndex(P(p), 0) * es), lda)

            JCublas2.cublasSswap(handle, N - i - realBlockSize, ADataPointer.withByteOffset(d_A.linearIndex(p, i + realBlockSize) * es), lda,
              ADataPointer.withByteOffset(d_A.linearIndex(P(p), i + realBlockSize) * es), lda)
          }

        }
        }

        JCublas2.cublasStrsm(handle, cublasSideMode.CUBLAS_SIDE_LEFT, cublasFillMode.CUBLAS_FILL_MODE_LOWER,
          cublasOperation.CUBLAS_OP_N, cublasDiagType.CUBLAS_DIAG_UNIT,
          realBlockSize, N - i - realBlockSize, onePtr,
          ADataPointer.withByteOffset(d_A.linearIndex(i, i) * es), lda,
          ADataPointer.withByteOffset(d_A.linearIndex(i, i + realBlockSize) * es), lda)


        if (i + realBlockSize < M) {
          JCublas2.cublasSgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
            M - i - realBlockSize, N - i - realBlockSize, realBlockSize, minusOnePtr,
            ADataPointer.withByteOffset(d_A.linearIndex(i + realBlockSize, i) * es), lda,
            ADataPointer.withByteOffset(d_A.linearIndex(i, i + realBlockSize) * es), lda,
            onePtr,
            ADataPointer.withByteOffset(d_A.linearIndex(i + realBlockSize, i + realBlockSize) * es), lda)
        }
      }
      }
    }

    LUBlockedFloat(d_A)
    // the pivoting matrix is Float (and not Int), because right now gust doesn't support
    // matrix multiplication other than Float*Float or Double*Double.
    // Of course, the swapping of rows is quite bad performance-wise but returning the
    // matrix and not a vector is a good thing, I guess

    // transforming permutation vector into permutation matrix
    // this could be done with a trick: cublasSetVector with incy = PM.majorStride + 1,
    // but I can't get CuMatrix.ones to work right now
    val PM = CuMatrix.fromDense(DenseMatrix.eye[Float](A.rows))

    cfor(0)(_ < PM.rows - 1, _ + 1) { i => {
      if (P(i) != i) {
        JCublas2.cublasSswap(handle, PM.cols,
          PM.offsetPointer.withByteOffset(PM.linearIndex(i, 0) * PM.elemSize), PM.majorSize,
          PM.offsetPointer.withByteOffset(PM.linearIndex(P(i), 0) * PM.elemSize), PM.majorSize)
      }
    }
    }

    (d_A, PM)
  }


  def LUDouble(A: CuMatrix[Double])(implicit handle: cublasHandle): (CuMatrix[Double], CuMatrix[Double]) = {
    val P: Array[Int] = (0 until A.rows).toArray
    val d_A = CuMatrix.create[Double](A.rows, A.cols)
    d_A := A
    val blockSize = 64
    val es = d_A.elemSize
    val lda = A.rows

    val one = Array(1.0)
    val onePtr = jcuda.Pointer.to(one)
    val minusOne = Array(-1.0)
    val minusOnePtr = jcuda.Pointer.to(minusOne)


    def LUSingleBlockDouble(M: Int, N: Int, ADataPointer: CuPointer, pOffset: Int): Unit = {

      val minDim = if (M < N) M else N
      val intRes = Array(0)
      val intResPtr = jcuda.Pointer.to(intRes)
      val A_ii = Array(0.0)
      val AiiPtr = jcuda.Pointer.to(A_ii)
      val alpha = Array(0.0)

      cfor(0)(_ < minDim, _ + 1) { i => {
        JCublas2.cublasIdamax(handle, M - i, ADataPointer.withByteOffset(d_A.linearIndex(i, i) * es),
          1, intResPtr)

        val pivotRow = i + intRes(0) - 1 // -1 because of cublas' 1-based indexing
        val ip1 = i + 1
        P(i + pOffset) = pivotRow + pOffset

        if (pivotRow != i) {
          // A.cols <-> N + uncomment in LUBlocked
          JCublas2.cublasDswap(handle, N, ADataPointer.withByteOffset(d_A.linearIndex(pivotRow, 0) * es),
            lda, ADataPointer.withByteOffset(d_A.linearIndex(i, 0) * es), lda)
        }

        JCuda.cudaMemcpy2D(AiiPtr, d_A.majorStride * d_A.elemSize,
          ADataPointer.withByteOffset(d_A.linearIndex(i, i) * es),
          d_A.majorStride * es, d_A.elemSize, 1, cudaMemcpyKind.cudaMemcpyDeviceToHost)
        // this one works as well:
        //      curesult =  JCublas2.cublasGetVector(1, es.toInt, ADataPointer.withByteOffset(d_A.linearIndex(i, i) * es),
        //                               1, AiiPtr, 1)

        if (Math.abs(A_ii(0)) < 1e-20) {
          println("Matrix is singular")
          return
        }

        if (ip1 < M) {
          alpha(0) = 1.0f / A_ii(0)
          JCublas2.cublasDscal(handle, M - ip1, jcuda.Pointer.to(alpha),
            ADataPointer.withByteOffset(d_A.linearIndex(ip1, i) * es), 1)
        }

        if (ip1 < minDim) {
          JCublas2.cublasDger(handle, M - ip1, N - ip1, minusOnePtr,
            ADataPointer.withByteOffset(d_A.linearIndex(ip1, i) * es), 1,
            ADataPointer.withByteOffset(d_A.linearIndex(i, ip1) * es), lda,
            ADataPointer.withByteOffset(d_A.linearIndex(ip1, ip1) * es), lda)

        }
      }
      }

      //d_A
    }

    def LUBlockedDouble(d_A: CuMatrix[Double]): Unit = {
      val M = d_A.rows
      val N = d_A.cols
      val minSize = if (M < N) M else N
      val ADataPointer = d_A.offsetPointer

      if (blockSize >= minSize) {
        LUSingleBlockDouble(M, N, ADataPointer, 0)
        return
      }

      cfor(0)(_ < minSize, _ + blockSize) { i => {
        val realBlockSize = if (minSize - i < blockSize) minSize - i else blockSize
        LUSingleBlockDouble(M - i, realBlockSize, ADataPointer.withByteOffset(d_A.linearIndex(i, i) * es), i)

        val mm = if (M < i + realBlockSize) M else i + realBlockSize

        cfor(i)(_ < mm - 1, _ + 1) { p => {
          // P(p) = P(p) + i // ???
          if (P(p) != p) {
            JCublas2.cublasDswap(handle, i, ADataPointer.withByteOffset(d_A.linearIndex(p, 0) * es),
              lda, ADataPointer.withByteOffset(d_A.linearIndex(P(p), 0) * es), lda)

            JCublas2.cublasDswap(handle, N - i - realBlockSize, ADataPointer.withByteOffset(d_A.linearIndex(p, i + realBlockSize) * es), lda,
              ADataPointer.withByteOffset(d_A.linearIndex(P(p), i + realBlockSize) * es), lda)
          }

        }
        }

        JCublas2.cublasDtrsm(handle, cublasSideMode.CUBLAS_SIDE_LEFT, cublasFillMode.CUBLAS_FILL_MODE_LOWER,
          cublasOperation.CUBLAS_OP_N, cublasDiagType.CUBLAS_DIAG_UNIT,
          realBlockSize, N - i - realBlockSize, onePtr,
          ADataPointer.withByteOffset(d_A.linearIndex(i, i) * es), lda,
          ADataPointer.withByteOffset(d_A.linearIndex(i, i + realBlockSize) * es), lda)


        if (i + realBlockSize < M) {
          JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
            M - i - realBlockSize, N - i - realBlockSize, realBlockSize, minusOnePtr,
            ADataPointer.withByteOffset(d_A.linearIndex(i + realBlockSize, i) * es), lda,
            ADataPointer.withByteOffset(d_A.linearIndex(i, i + realBlockSize) * es), lda,
            onePtr,
            ADataPointer.withByteOffset(d_A.linearIndex(i + realBlockSize, i + realBlockSize) * es), lda)
        }
      }
      }
    }

    LUBlockedDouble(d_A)

    val PM = CuMatrix.fromDense(DenseMatrix.eye[Double](A.rows))

    cfor(0)(_ < PM.rows - 1, _ + 1) { i => {
      if (P(i) != i) {
        JCublas2.cublasSswap(handle, PM.cols,
          PM.offsetPointer.withByteOffset(PM.linearIndex(i, 0) * PM.elemSize), PM.majorSize,
          PM.offsetPointer.withByteOffset(PM.linearIndex(P(i), 0) * PM.elemSize), PM.majorSize)
      }
    }
    }

    (d_A, PM)
  }


  /**
   * LU decomposition using the cublas(S|D)getrfBatched -- not very fast
   * and works only for square matrices
   * @param A
   * @param handle
   * @return
   */
  def LUFloatCublas(A: CuMatrix[Float])(implicit handle: cublasHandle): (CuMatrix[Float], CuMatrix[Float]) = {

    if (A.rows != A.cols) {
      println("A has to be a square matrix.")
      return (A, CuMatrix.fromDense(DenseMatrix.eye[Float](A.rows))) // got to return something, perhaps
      // an eye is better than null
    }

    val d_A = CuMatrix.create[Float](A.rows, A.cols)
    d_A := A

    val P = CuMatrix.create[Int](d_A.rows, 1)
    val info = CuMatrix.create[Int](1, 1)

    val A_ptr = jcuda.Pointer.to(d_A.offsetPointer)
    val d_Aptr = new jcuda.Pointer()

    JCuda.cudaMalloc(d_Aptr, jcuda.Sizeof.POINTER)
    JCuda.cudaMemcpy(d_Aptr, A_ptr, 1 * jcuda.Sizeof.POINTER, cudaMemcpyKind.cudaMemcpyHostToDevice)

    JCublas2.cublasSgetrfBatched(handle, d_A.rows, d_Aptr, d_A.majorSize,
      P.offsetPointer, info.offsetPointer, 1)

    // transforming permutation vector into permutation matrix
    val h_P = P.toDense
    val PM = CuMatrix.fromDense(DenseMatrix.eye[Float](A.rows))

    cfor(0)(_ < PM.rows - 1, _ + 1) { i => {
      if (h_P(i, 0) != i + 1) {
        // the returned vector uses 1-based indexing
        JCublas2.cublasSswap(handle, PM.cols,
          PM.offsetPointer.withByteOffset(PM.linearIndex(i, 0) * PM.elemSize), PM.majorSize,
          PM.offsetPointer.withByteOffset(PM.linearIndex(h_P(i, 0) - 1, 0) * PM.elemSize), PM.majorSize)
      }
    }
    }

    (d_A, PM)
  }

  def LUDoubleCublas(A: CuMatrix[Double])(implicit handle: cublasHandle): (CuMatrix[Double], CuMatrix[Double]) = {

    if (A.rows != A.cols) {
      println("A has to be a square matrix.")
      return (A, CuMatrix.fromDense(DenseMatrix.eye[Double](A.rows)))
    }

    val d_A = CuMatrix.create[Double](A.rows, A.cols)
    d_A := A

    val P = CuMatrix.create[Int](d_A.rows, 1)
    val info = CuMatrix.create[Int](1, 1)

    val A_ptr = jcuda.Pointer.to(d_A.offsetPointer)
    val d_Aptr = new jcuda.Pointer()

    JCuda.cudaMalloc(d_Aptr, jcuda.Sizeof.POINTER)
    JCuda.cudaMemcpy(d_Aptr, A_ptr, 1 * jcuda.Sizeof.POINTER, cudaMemcpyKind.cudaMemcpyHostToDevice)

    JCublas2.cublasDgetrfBatched(handle, d_A.rows, d_Aptr, d_A.majorSize,
      P.offsetPointer, info.offsetPointer, 1)

    val h_P = P.toDense
    val PM = CuMatrix.fromDense(DenseMatrix.eye[Double](A.rows))

    cfor(0)(_ < PM.rows - 1, _ + 1) { i => {
      if (h_P(i, 0) != i + 1) {
        // the returned vector uses 1-based indexing
        JCublas2.cublasDswap(handle, PM.cols,
          PM.offsetPointer.withByteOffset(PM.linearIndex(i, 0) * PM.elemSize), PM.majorSize,
          PM.offsetPointer.withByteOffset(PM.linearIndex(h_P(i, 0) - 1, 0) * PM.elemSize), PM.majorSize)
      }
    }
    }

    (d_A, PM)
  }

}