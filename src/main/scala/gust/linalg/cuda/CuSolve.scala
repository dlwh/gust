package gust.linalg.cuda

import gust.linalg.cuda.CuWrapperMethods._
import jcuda.jcublas._
import jcuda.jcusparse._
import spire.syntax.cfor._

import scala.annotation.tailrec


/**
 * Copyright 2014 Piotr Moczurad
 *
 */
object CuSolve {

  /**
   * LU solve for square matrices:
   * First we compute P, L, U such that: PA = LU
   * Next, we solve LY = PB and then solve UX = Y
   * @param A a square n-by-n matrix
   * @param b column vector (of length n)
   * @return column vector of length n containing the solution
   */
  def LUSolveFloat(A: CuMatrix[Float], b: CuMatrix[Float])(implicit handle: cublasHandle): CuMatrix[Float] = {
    if (A.rows != A.cols) {
      println("A has to be a square matrix")
      return b
    }

    val n = A.rows
    val oneArr = Array(1.0f)
    val zeroArr = Array(0.0f)

    val (d_A, d_P) = CuLU.LUFloat(A)
    val (d_L, d_U) = CuLU.LUFactorsFloat(d_A)
    val d_b = CuMatrix.create[Float](b.rows, b.cols); d_b := b

    // d_b = d_P * d_b
    SgemmNN(n, 1, n, jcuda.Pointer.to(oneArr), d_P, 0,0, d_b, 0,0, jcuda.Pointer.to(zeroArr), d_b, 0,0)

    // solve: d_L * Y = d_b and store Y inside d_b:
    JCublas2.cublasStrsv(handle, cublasFillMode.CUBLAS_FILL_MODE_LOWER, cublasOperation.CUBLAS_OP_N,
      cublasDiagType.CUBLAS_DIAG_UNIT, d_L.cols, d_L.offsetPointer, d_L.majorStride, d_b.offsetPointer, 1)

    // solve d_U * X = d_b and store X inside d_b:
    JCublas2.cublasStrsv(handle, cublasFillMode.CUBLAS_FILL_MODE_UPPER, cublasOperation.CUBLAS_OP_N,
      cublasDiagType.CUBLAS_DIAG_NON_UNIT, d_U.cols, d_U.offsetPointer, d_U.majorStride, d_b.offsetPointer, 1)

    d_b
  }

  def LUSolveDouble(A: CuMatrix[Double], b: CuMatrix[Double])(implicit handle: cublasHandle): CuMatrix[Double] = {
    if (A.rows != A.cols) {
      println("A has to be a square matrix")
      return b
    }

    val n = A.rows
    val oneArr = Array(1.0)
    val zeroArr = Array(0.0)

    val (d_A, d_P) = CuLU.LUDouble(A)
    val (d_L, d_U) = CuLU.LUFactorsDouble(d_A)
    val d_b = CuMatrix.create[Double](b.rows, b.cols); d_b := b

    // d_b = d_P * d_b
    DgemmNN(n, 1, n, jcuda.Pointer.to(oneArr), d_P, 0,0, d_b, 0,0, jcuda.Pointer.to(zeroArr), d_b, 0,0)

    // solve: d_L * Y = d_b and store Y inside d_b:
    JCublas2.cublasDtrsv(handle, cublasFillMode.CUBLAS_FILL_MODE_LOWER, cublasOperation.CUBLAS_OP_N,
      cublasDiagType.CUBLAS_DIAG_UNIT, d_L.cols, d_L.offsetPointer, d_L.majorStride, d_b.offsetPointer, 1)

    // solve d_U * X = d_b and store X inside d_b:
    JCublas2.cublasDtrsv(handle, cublasFillMode.CUBLAS_FILL_MODE_UPPER, cublasOperation.CUBLAS_OP_N,
      cublasDiagType.CUBLAS_DIAG_NON_UNIT, d_U.cols, d_U.offsetPointer, d_U.majorStride, d_b.offsetPointer, 1)

    d_b
  }

  /**
   * QR solve for matrices that are not necessarily square.
   * The idea is like this (using matlab's notation):
   * We want to solve Ax = b
   * compute QR = A
   * now QRx = b, multiply both sides by Q**-1 (which is Q')
   * we get Rx = Q' * b
   *    [Q, R] = qr(A);
   *    x = R \ (Q' * b);
   *
   * @param A (dimensions: m x n with m >= n)
   * @param b
   * @return
   */
  def QRSolveFloat(A: CuMatrix[Float], b: CuMatrix[Float])(implicit handle: cublasHandle): CuMatrix[Float] = {
    if (A.rows < A.cols) {
      println("Number of rows of matrix A cannot be smaller than the number of columns.")
      return b
    }

    // [Q, R] = qr(A)
    val (d_A, tau) = CuQR.QRFloatMN(A)
    val (d_Q, d_R) = CuQR.QRFactorsFloat(d_A, tau)

    val d_Qtb = d_Q.t * b   // Q' * b

    // solve the triangular system Rx = Q' * b
    JCublas2.cublasStrsv(handle, cublasFillMode.CUBLAS_FILL_MODE_UPPER, cublasOperation.CUBLAS_OP_N,
      cublasDiagType.CUBLAS_DIAG_NON_UNIT, d_R.cols, d_R.offsetPointer, d_R.majorStride, d_Qtb.offsetPointer, 1)

    d_Qtb(0 until A.cols, 0)
  }

  def QRSolveDouble(A: CuMatrix[Double], b: CuMatrix[Double])(implicit handle: cublasHandle): CuMatrix[Double] = {
    if (A.rows < A.cols) {
      println("Number of rows of matrix A cannot be smaller than the number of columns.")
      return b
    }

    // [Q, R] = qr(A)
    val (d_A, tau) = CuQR.QRDoubleMN(A)
    val (d_Q, d_R) = CuQR.QRFactorsDouble(d_A, tau)

    val d_Qtb = d_Q.t * b   // Q' * b

    // solve the triangular system Rx = Q' * b
    JCublas2.cublasDtrsv(handle, cublasFillMode.CUBLAS_FILL_MODE_UPPER, cublasOperation.CUBLAS_OP_N,
      cublasDiagType.CUBLAS_DIAG_NON_UNIT, d_R.cols, d_R.offsetPointer, d_R.majorStride, d_Qtb.offsetPointer, 1)

    d_Qtb(0 until A.cols, 0)
  }


  /* sparse solves */

  /**
   * Sparse linear equation solver using the basic Gauss-Seidel (SR) method.
   * @param A coefficent matrix
   * @param b right hand side
   * @return solution vector
   */
  def sparseSolveFloat(A: CuSparseMatrix[Float], b: CuMatrix[Float])(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle): CuMatrix[Float] = {
    // construct L (lower triangular part of matrix A) and U (strictly upper triangular part)
    val denseCsrValL = A.cscVal.toDense
    val denseCsrValU = A.cscVal.toDense
    val denseCsrRowPtrA = A.cscColPtr.toDense
    val denseCsrColIndA = A.cscRowInd.toDense

    // construct L (ones on the diagonal) and U:
    cfor(0)(_ < A.cols, _ + 1) { i => {
      cfor(denseCsrRowPtrA(i, 0))(_ < denseCsrRowPtrA(i+1, 0), _ + 1) { j => {
        val row = i
        val col = denseCsrColIndA(j, 0)

        if (row > col) denseCsrValL(j, 0) = 0.0f
        if (row == col || row < col) denseCsrValU(j, 0) = 0.0f
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

    val L = new CuSparseMatrix[Float](A.rows, A.cols, descrL, CuMatrix.fromDense(denseCsrValL),
      CuMatrix.fromDense(denseCsrRowPtrA), CuMatrix.fromDense(denseCsrColIndA))

    val U = new CuSparseMatrix[Float](A.rows, A.cols, descrU, CuMatrix.fromDense(denseCsrValU),
        CuMatrix.fromDense(denseCsrRowPtrA), CuMatrix.fromDense(denseCsrColIndA))

    // reverse the sign of U:
    JCublas2.cublasSscal(blasHandle, U.nnz, jcuda.Pointer.to(Array(-1.0f)), U.cscVal.offsetPointer, 1)

    // create the initial guess (all zeros for now):
    val x_0 = CuMatrix.create[Float](A.rows, 1)

    // and let the Gauss-Seidel iteration do its thing:
    gsIterFloat(L, U, b, x_0, 10*A.rows*A.rows) // the number of iterations here is basically a guess but seems to work
  }

  def sparseSolveDouble(A: CuSparseMatrix[Double], b: CuMatrix[Double])(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle): CuMatrix[Double] = {
    // construct L (lower triangular part of matrix A) and U (strictly upper triangular part)
    val denseCsrValL = A.cscVal.toDense
    val denseCsrValU = A.cscVal.toDense
    val denseCsrRowPtrA = A.cscColPtr.toDense
    val denseCsrColIndA = A.cscRowInd.toDense

    // construct L (ones on the diagonal) and U:
    cfor(0)(_ < A.cols, _ + 1) { i => {
      cfor(denseCsrRowPtrA(i, 0))(_ < denseCsrRowPtrA(i+1, 0), _ + 1) { j => {
        val row = i
        val col = denseCsrColIndA(j, 0)

        if (row > col) denseCsrValL(j, 0) = 0.0
        if (row == col || row < col) denseCsrValU(j, 0) = 0.0
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

    val L = new CuSparseMatrix[Double](A.rows, A.cols, descrL, CuMatrix.fromDense(denseCsrValL),
      CuMatrix.fromDense(denseCsrRowPtrA), CuMatrix.fromDense(denseCsrColIndA))

    val U = new CuSparseMatrix[Double](A.rows, A.cols, descrU, CuMatrix.fromDense(denseCsrValU),
      CuMatrix.fromDense(denseCsrRowPtrA), CuMatrix.fromDense(denseCsrColIndA))

    // reverse the sign of U:
    JCublas2.cublasDscal(blasHandle, U.nnz, jcuda.Pointer.to(Array(-1.0)), U.cscVal.offsetPointer, 1)

    // create the initial guess (all zeros for now):
    val x_0 = CuMatrix.create[Double](A.rows, 1)

    gsIterDouble(L, U, b, x_0, 10*A.rows*A.rows)
  }

  /**
   * Performs subsequent iterations of Gauss-Seidel method.
   * Returns the vector x_k+1
   * The iteration step is as follows:  x_k+1 = inv(L)*U*x_k + inv(L)*b
   * This actually means solving:       L*x_k+1 = U*x_k + b
   * @param L lower triangular part of the matrix
   * @param U strictly upper triangular part of the matrix, with sign inverted
   * @param b right hand side
   * @param x_k initial approximation of the solution
   * @param maxIter max number of iterations
   * @return x_k+1, the improved approximation of the solution
   */
  @tailrec
  private def gsIterFloat(L: CuSparseMatrix[Float], U: CuSparseMatrix[Float], b: CuMatrix[Float], x_k: CuMatrix[Float], maxIter: Int)(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle): CuMatrix[Float] = {
    val m = L.rows
    val n = L.cols

    // calculate the right hand side:
    val c: CuMatrix[Float] = U * x_k //+ b
    c += b

    // solve the triangular system:
    val x_k_1 = sparseSolveTriFloat(L, c, 'L')

    if ((x_k_1 - x_k).norm < 1e-2 || maxIter <= 1) x_k_1
    else gsIterFloat(L, U, b, x_k_1, maxIter-1)
  }

  @tailrec
  private def gsIterDouble(L: CuSparseMatrix[Double], U: CuSparseMatrix[Double], b: CuMatrix[Double], x_k: CuMatrix[Double], maxIter: Int)(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle): CuMatrix[Double] = {
    val m = L.rows
    val n = L.cols
    val one = jcuda.Pointer.to(Array(1.0))
    val minusOne = jcuda.Pointer.to(Array(-1.0))

    // calculate the right hand side:
    val c: CuMatrix[Double] = U * x_k //+ b
    JCublas2.cublasDaxpy(blasHandle, c.size, one, b.offsetPointer, 1, c.offsetPointer, 1)

    // solve the triangular system:
    val x_k_1 = sparseSolveTriDouble(L, c, 'L')
    val x_diff = CuMatrix.create[Double](x_k_1.rows, 1)
    JCublas2.cublasDaxpy(blasHandle, x_diff.rows, minusOne, x_k_1.offsetPointer, 1, x_diff.offsetPointer, 1)

    if (x_diff.norm < 1e-2 || maxIter <= 1) x_k_1
    else gsIterDouble(L, U, b, x_k_1, maxIter-1)
  }


  def sparseSolveTriFloat(A: CuSparseMatrix[Float], b: CuMatrix[Float], fillMode: Char = 'U')(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle): CuMatrix[Float] = {
    val AS = A.transpose // csc2csr

    if (AS.rows != AS.cols) {
      println("A has to be a square matrix")
      return b
    }

    val info = new cusparseSolveAnalysisInfo
    JCusparse2.cusparseCreateSolveAnalysisInfo(info)
    val trans = if (AS.isTranspose) cusparseOperation.CUSPARSE_OPERATION_TRANSPOSE else cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE
    if (fillMode == 'L')
      JCusparse2.cusparseSetMatFillMode(AS.descr, cusparseFillMode.CUSPARSE_FILL_MODE_LOWER)
    else
      JCusparse2.cusparseSetMatFillMode(AS.descr, cusparseFillMode.CUSPARSE_FILL_MODE_UPPER)

    val m = AS.rows
    val nnz = AS.cscVal.rows
    val one = jcuda.Pointer.to(Array(1.0f))
    val res = CuMatrix.create[Float](m, 1)  // vector that will contain the solution
    JCusparse2.cusparseScsrsv_analysis(sparseHandle, trans, m, nnz, AS.descr, AS.cscVal.offsetPointer, AS.cscColPtr.offsetPointer, AS.cscRowInd.offsetPointer, info)
    JCusparse2.cusparseScsrsv_solve(sparseHandle, trans, m, one, AS.descr, AS.cscVal.offsetPointer, AS.cscColPtr.offsetPointer, AS.cscRowInd.offsetPointer, info, b.offsetPointer, res.offsetPointer)

    JCusparse2.cusparseDestroySolveAnalysisInfo(info)

    res
  }

  def sparseSolveTriDouble(A: CuSparseMatrix[Double], b: CuMatrix[Double], fillMode: Char = 'U')(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle): CuMatrix[Double] = {
    val AS = A.transpose // csc2csr

    if (AS.rows != AS.cols) {
      println("A has to be a square matrix")
      return b
    }

    val info = new cusparseSolveAnalysisInfo
    JCusparse2.cusparseCreateSolveAnalysisInfo(info)
    val trans = if (AS.isTranspose) cusparseOperation.CUSPARSE_OPERATION_TRANSPOSE else cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE
    if (fillMode == 'L')
      JCusparse2.cusparseSetMatFillMode(AS.descr, cusparseFillMode.CUSPARSE_FILL_MODE_LOWER)
    else
      JCusparse2.cusparseSetMatFillMode(AS.descr, cusparseFillMode.CUSPARSE_FILL_MODE_UPPER)

    val m = AS.rows
    val nnz = AS.cscVal.rows
    val one = jcuda.Pointer.to(Array(1.0))
    val res = CuMatrix.create[Double](m, 1)  // vector that will contain the solution
    JCusparse2.cusparseDcsrsv_analysis(sparseHandle, trans, m, nnz, AS.descr, AS.cscVal.offsetPointer, AS.cscColPtr.offsetPointer, AS.cscRowInd.offsetPointer, info)
    JCusparse2.cusparseDcsrsv_solve(sparseHandle, trans, m, one, AS.descr, AS.cscVal.offsetPointer, AS.cscColPtr.offsetPointer, AS.cscRowInd.offsetPointer, info, b.offsetPointer, res.offsetPointer)

    JCusparse2.cusparseDestroySolveAnalysisInfo(info)

    res
  }

}
