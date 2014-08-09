package gust.linalg.cuda

import breeze.linalg.support.{CanCollapseAxis, CanTranspose, CanSlice2}


import jcuda.jcublas.{cublasOperation, cublasHandle, JCublas2, cublasDiagType, cublasFillMode, cublasSideMode}
import gust.util.cuda
import jcuda.jcusparse._
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
    val (d_A, h_P) = LUFloatSimplePivot(A)

    val PM = CuMatrix.fromDense(DenseMatrix.eye[Float](A.rows))

    cfor(0)(_ < PM.rows - 1, _ + 1) { i => {
      if (h_P(i) != i) {
        JCublas2.cublasSswap(handle, PM.cols,
          PM.offsetPointer.withByteOffset(PM.linearIndex(i, 0) * PM.elemSize), PM.majorSize,
          PM.offsetPointer.withByteOffset(PM.linearIndex(h_P(i), 0) * PM.elemSize), PM.majorSize)
      }
    }
    }

    (d_A, PM)

  }

  /**
   * This returns the vector with row interchanges (it makes the determinant easier to implement)
   * and that's how Breeze does it
   * @param A
   * @return matrix with both L and U, array with pivoting values
   */
  def LUFloatSimplePivot(A: CuMatrix[Float])(implicit handle: cublasHandle): (CuMatrix[Float], Array[Int]) = {
    val P: Array[Int] = (0 until A.rows).toArray
    val d_A = CuMatrix.create[Float](A.rows, A.cols)
    d_A := A
    val blockSize = 32
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

    (d_A, P)
  }


  def LUDouble(A: CuMatrix[Double])(implicit handle: cublasHandle): (CuMatrix[Double], CuMatrix[Double]) = {
    val (d_A, h_P) = LUDoubleSimplePivot(A)

    val PM = CuMatrix.fromDense(DenseMatrix.eye[Double](A.rows))

    cfor(0)(_ < PM.rows - 1, _ + 1) { i => {
      if (h_P(i) != i) {
        JCublas2.cublasDswap(handle, PM.cols,
          PM.offsetPointer.withByteOffset(PM.linearIndex(i, 0) * PM.elemSize), PM.majorSize,
          PM.offsetPointer.withByteOffset(PM.linearIndex(h_P(i), 0) * PM.elemSize), PM.majorSize)
      }
    }
    }

    (d_A, PM)
  }

  def LUDoubleSimplePivot(A: CuMatrix[Double])(implicit handle: cublasHandle): (CuMatrix[Double], Array[Int]) = {
    val P: Array[Int] = (0 until A.rows).toArray
    val d_A = CuMatrix.create[Double](A.rows, A.cols)
    d_A := A
    val blockSize = 32
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
          //P(p) = P(p) + i // ???
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

    (d_A, P)
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

  def incompleteLUFloat(A: CuSparseMatrix[Float])(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle): CuSparseMatrix[Float] = {
    if (A.rows != A.cols) {
      println("A has to be a square matrix")
      return A
    }

    val AS = A.copy

    val info = new cusparseSolveAnalysisInfo
    JCusparse2.cusparseCreateSolveAnalysisInfo(info)
    val trans = if (AS.isTranspose) cusparseOperation.CUSPARSE_OPERATION_TRANSPOSE else cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE
    val m = AS.rows
    JCusparse2.cusparseScsrsv_analysis(sparseHandle, trans, m, AS.nnz, AS.descr, AS.cscVal.offsetPointer, AS.cscColPtr.offsetPointer, AS.cscRowInd.offsetPointer, info)
    JCusparse2.cusparseScsrilu0(sparseHandle, trans, m, AS.descr, AS.cscVal.offsetPointer, AS.cscColPtr.offsetPointer, AS.cscRowInd.offsetPointer, info)

    JCusparse2.cusparseDestroySolveAnalysisInfo(info)

    AS
  }

  def incompleteLUDouble(A: CuSparseMatrix[Double])(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle): CuSparseMatrix[Double] = {
    if (A.rows != A.cols) {
      println("A has to be a square matrix")
      return A
    }

    val AS = A.copy

    val info = new cusparseSolveAnalysisInfo
    JCusparse2.cusparseCreateSolveAnalysisInfo(info)
    val trans = if (AS.isTranspose) cusparseOperation.CUSPARSE_OPERATION_TRANSPOSE else cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE
    val m = AS.rows
    JCusparse2.cusparseDcsrsv_analysis(sparseHandle, trans, m, AS.nnz, AS.descr, AS.cscVal.offsetPointer, AS.cscColPtr.offsetPointer, AS.cscRowInd.offsetPointer, info)
    JCusparse2.cusparseDcsrilu0(sparseHandle, trans, m, AS.descr, AS.cscVal.offsetPointer, AS.cscColPtr.offsetPointer, AS.cscRowInd.offsetPointer, info)

    JCusparse2.cusparseDestroySolveAnalysisInfo(info)

    AS
  }

  def incompleteLUFactorsFloat(A: CuSparseMatrix[Float])(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle): (CuSparseMatrix[Float], CuSparseMatrix[Float]) = {
    val denseCsrValL = A.cscVal.toDense
    val denseCsrValU = A.cscVal.toDense
    val denseCsrRowPtrA = A.cscColPtr.toDense
    val denseCsrColIndA = A.cscRowInd.toDense

    // construct L (ones on the diagonal) and U:
    cfor(0)(_ < A.rows, _ + 1) { i => {
      cfor(denseCsrRowPtrA(i, 0))(_ < denseCsrRowPtrA(i+1, 0), _ + 1) { j => {
        val row = i
        val col = denseCsrColIndA(j, 0)

        if (row == col) denseCsrValL(j, 0) = 1.0f
        else if (row < col) denseCsrValL(j, 0) = 0.0f

        if (row > col) denseCsrValU(j, 0) = 0.0f
      }}
    }}

    // create new descriptors:
    val descrL = new cusparseMatDescr
    JCusparse2.cusparseCreateMatDescr(descrL)
    JCusparse2.cusparseSetMatType(descrL, cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL)
    //JCusparse2.cusparseSetMatFillMode(descrL, cusparseFillMode.CUSPARSE_FILL_MODE_LOWER)
    JCusparse2.cusparseSetMatIndexBase(descrL, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO)
    JCusparse2.cusparseSetMatDiagType(descrL, cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT)

    val descrU = new cusparseMatDescr
    JCusparse2.cusparseCreateMatDescr(descrU)
    JCusparse2.cusparseSetMatType(descrU, cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL)
    //JCusparse2.cusparseSetMatFillMode(descrU, cusparseFillMode.CUSPARSE_FILL_MODE_UPPER)
    JCusparse2.cusparseSetMatIndexBase(descrU, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO)
    JCusparse2.cusparseSetMatDiagType(descrU, cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT)

    (new CuSparseMatrix[Float](A.rows, A.cols, descrL, CuMatrix.fromDense(denseCsrValL),
                           CuMatrix.fromDense(denseCsrRowPtrA), CuMatrix.fromDense(denseCsrColIndA)).transpose,
     new CuSparseMatrix[Float](A.rows, A.cols, descrU, CuMatrix.fromDense(denseCsrValU),
                           CuMatrix.fromDense(denseCsrRowPtrA), CuMatrix.fromDense(denseCsrColIndA)).transpose)
  }

  def incompleteLUFactorsDouble(A: CuSparseMatrix[Double])(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle): (CuSparseMatrix[Double], CuSparseMatrix[Double]) = {
    val denseCsrValL = A.cscVal.toDense
    val denseCsrValU = A.cscVal.toDense
    val denseCsrRowPtrA = A.cscColPtr.toDense
    val denseCsrColIndA = A.cscRowInd.toDense

    // construct L (ones on the diagonal) and U:
    cfor(0)(_ < A.rows, _ + 1) { i => {
      cfor(denseCsrRowPtrA(i, 0))(_ < denseCsrRowPtrA(i+1, 0), _ + 1) { j => {
        val row = i
        val col = denseCsrColIndA(j, 0)

        if (row == col) denseCsrValL(j, 0) = 1.0f
        else if (row < col) denseCsrValL(j, 0) = 0.0f

        if (row > col) denseCsrValU(j, 0) = 0.0f
      }}
    }}

    // create new descriptors:
    val descrL = new cusparseMatDescr
    JCusparse2.cusparseCreateMatDescr(descrL)
    JCusparse2.cusparseSetMatType(descrL, cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL)
    //JCusparse2.cusparseSetMatFillMode(descrL, cusparseFillMode.CUSPARSE_FILL_MODE_LOWER)
    JCusparse2.cusparseSetMatIndexBase(descrL, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO)
    JCusparse2.cusparseSetMatDiagType(descrL, cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT)

    val descrU = new cusparseMatDescr
    JCusparse2.cusparseCreateMatDescr(descrU)
    JCusparse2.cusparseSetMatType(descrU, cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL)
    //JCusparse2.cusparseSetMatFillMode(descrU, cusparseFillMode.CUSPARSE_FILL_MODE_UPPER)
    JCusparse2.cusparseSetMatIndexBase(descrU, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO)
    JCusparse2.cusparseSetMatDiagType(descrU, cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT)

    (new CuSparseMatrix[Double](A.rows, A.cols, descrL, CuMatrix.fromDense(denseCsrValL),
      CuMatrix.fromDense(denseCsrRowPtrA), CuMatrix.fromDense(denseCsrColIndA)).transpose,
      new CuSparseMatrix[Double](A.rows, A.cols, descrU, CuMatrix.fromDense(denseCsrValU),
        CuMatrix.fromDense(denseCsrRowPtrA), CuMatrix.fromDense(denseCsrColIndA)).transpose)
  }


  /**
   * LU decomposition based on gpu_sgetrf code by V. Volkov.
   * @param A
   * @param handle
   * @return
   */
  def LUSPFloat(A: CuMatrix[Float])(implicit handle: cublasHandle): (CuMatrix[Float], Array[Int]) = {
    val nb = 64

    val m = A.rows
    val n = A.cols
    val lda = A.majorStride

    val one = Pointer.pointerToFloat(1.0f).toCuPointer
    val zero = Pointer.pointerToFloat(0.0f).toCuPointer
    val minusOne = Pointer.pointerToFloat(-1.0f).toCuPointer

    val gpu_matrix = CuMatrix.zeros[Float](m, n); //gpu_matrix := A
    val gpu_buff = CuMatrix.zeros[Float](n, nb)
    val gpu_L = CuMatrix.zeros[Float](nb, nb)

    val cpu_matrix = A.toDense
    val cpu_L = new DenseMatrix[Float](nb, nb)
    val cpu_p = new Array[Int](m)

    val info = new intW(0)

    uploadFloat(n, n-nb, gpu_matrix, 0, nb, cpu_matrix, 0, nb)
    transposeInplaceFloat(n, n, gpu_matrix, 0, 0)

    cfor(0)(_ < n, _ + nb) { i => {
      val h = n - i
      val w = if (h < nb) h else nb

      if (i > 0) {
        transposeFloat(h, w, gpu_buff, 0, 0, gpu_matrix, i, i) // !!
        downloadFloat(h, w, cpu_matrix, i, i, gpu_buff, 0, 0)

        JCublas2.cublasStrsm(handle, cublasSideMode.CUBLAS_SIDE_RIGHT, cublasFillMode.CUBLAS_FILL_MODE_UPPER,
          cublasOperation.CUBLAS_OP_N, cublasDiagType.CUBLAS_DIAG_UNIT, h-nb, nb, one,
          gpu_matrix.offsetPointer.withByteOffset(gpu_matrix.linearIndex(i-nb, i-nb) * gpu_matrix.elemSize), gpu_matrix.majorStride,
          gpu_matrix.offsetPointer.withByteOffset(gpu_matrix.linearIndex(i+nb, i-nb) * gpu_matrix.elemSize), gpu_matrix.majorStride)

        SgemmNN(h-nb, h, nb, minusOne, gpu_matrix, i+nb, i-nb, gpu_matrix, i-nb, i, one, gpu_matrix, i+nb, i)
      }

      println("majorStride: " + cpu_matrix.majorStride + "  i+i*cpu_matrix.majorStride: " + (i+i*cpu_matrix.majorStride))
      lapack.sgetrf(h, w, cpu_matrix.data, cpu_matrix.linearIndex(i, i), cpu_matrix.majorStride, cpu_p, i, info)

      batchSwapFloat(w, n, cpu_p, i, gpu_matrix, 0, i)
      cfor(0)(_ < w, _ + 1) { j => {
        cpu_p(i+j) += i
      }}

      uploadFloat(h, w, gpu_buff, 0, 0, cpu_matrix, i, i)
      transposeFloat(w, h, gpu_matrix, i, i, gpu_buff, 0, 0)

      if (h > nb) {
        JCublas2.cublasStrsm(handle, cublasSideMode.CUBLAS_SIDE_RIGHT, cublasFillMode.CUBLAS_FILL_MODE_UPPER,
          cublasOperation.CUBLAS_OP_N, cublasDiagType.CUBLAS_DIAG_UNIT, nb, nb, one,
          gpu_matrix.offsetPointer.withByteOffset(gpu_matrix.linearIndex(i, i) * gpu_matrix.elemSize), gpu_matrix.majorStride,
          gpu_matrix.offsetPointer.withByteOffset(gpu_matrix.linearIndex(i+nb, i) * gpu_matrix.elemSize), gpu_matrix.majorStride)

        SgemmNN(nb, h-nb, nb, minusOne, gpu_matrix, i+nb, i, gpu_matrix, i, i+nb, one, gpu_matrix, i+nb, i+nb)
      }

    }}

    transposeInplaceFloat(n, n, gpu_matrix, 0, 0)

    (gpu_matrix, cpu_p map { _ - 1})
  }

  def pivotMatrixFloat(A: CuMatrix[Float], cpu_p: Array[Int])(implicit handle: cublasHandle) {
    cfor(0)(_ < cpu_p.length, _ + 1) { i => {
      if (i != cpu_p(i)) {
        JCublas2.cublasSswap(handle, A.cols, A.offsetPointer.withByteOffset(A.linearIndex(i, 0) * A.elemSize), A.majorStride,
          A.offsetPointer.withByteOffset(A.linearIndex(cpu_p(i), 0) * A.elemSize), A.majorStride)
      }
    }}
  }

  def pivotMatrixDouble(A: CuMatrix[Double], cpu_p: Array[Int])(implicit handle: cublasHandle) {
    cfor(0)(_ < cpu_p.length, _ + 1) { i => {
      if (i != cpu_p(i)) {
        JCublas2.cublasDswap(handle, A.cols, A.offsetPointer.withByteOffset(A.linearIndex(i, 0) * A.elemSize), A.majorStride,
          A.offsetPointer.withByteOffset(A.linearIndex(cpu_p(i), 0) * A.elemSize), A.majorStride)
      }
    }}
  }
}