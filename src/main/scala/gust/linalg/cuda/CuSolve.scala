package gust.linalg.cuda

import gust.linalg.cuda.CuWrapperMethods._
import jcuda.jcublas._


/**
 * Created by piotrek on 05.07.2014.
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




}
