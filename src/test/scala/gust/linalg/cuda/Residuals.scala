package gust.linalg.cuda

import breeze.linalg.{convert, DenseMatrix}
import jcuda.jcublas.{JCublas2, cublasHandle, cublasOperation}
import org.bridj.Pointer
import gust.util.cuda._

/**
 * Created by Piotr on 2014-08-06.
 *
 * Methods for calculating residuals for the factorization methods.
 */
object Residuals {

  val zeroPtr = Pointer.pointerToDouble(0.0)
  val zero = zeroPtr.toCuPointer
  val onePtr = Pointer.pointerToDouble(1.0)
  val one = onePtr.toCuPointer

  def LUResiduals(implicit handle: cublasHandle) {
    val mats = getMatrices(32)
    mats foreach { mat => {
      val mat_copy = CuMatrix.create[Double](mat.rows, mat.cols); mat_copy := mat
      val (d_A, h_p) = CuLU.LUDoubleSimplePivot(mat_copy)
      val (d_L, d_U) = CuLU.LUFactorsDouble(d_A)
      CuLU.pivotMatrixDouble(mat_copy, h_p)
      val r = CuWrapperMethods.residualDouble(mat_copy, d_L, d_U, null)
      println("    " + mat_copy.rows + "    " + r)
      d_A.release()
      d_L.release()
      d_U.release()
      mat.release()
      mat_copy.release()
    }}
  }

  def QRResiduals(implicit handle: cublasHandle) {
    val mats = getMatrices(32)
    mats foreach { mat => {
      val (d_A, tau) = CuQR.QRDoubleMN(mat)
      val (d_Q, d_R) = CuQR.QRFactorsDouble(d_A, tau)
      val r = CuWrapperMethods.residualDouble(mat, d_Q, d_R, null)
      println("    " + mat.rows + "    " + r)
      d_A.release()
      d_Q.release()
      d_R.release()
      mat.release()
    }}
  }

  def CholeskyResiduals(implicit handle: cublasHandle) {
    val mats = getMatrices(32)
    mats foreach { mat => {
      // symmetric matrix:
      // TODO make it SPD
      val d_A = CuMatrix.create[Double](mat.rows, mat.cols); d_A := mat
      CuWrapperMethods.zeroOutDouble(d_A, 'U')
      JCublas2.cublasDgeam(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T, mat.rows, mat.cols,
        one, d_A.offsetPointer, d_A.majorStride, one, d_A.offsetPointer, d_A.majorStride, mat.offsetPointer, mat.majorStride)

      val d_L = CuCholesky.choleskyDouble(d_A)
      val r = CuWrapperMethods.residualDouble(mat, d_L, d_L.t, null)
      println("    " + mat.rows + "    " + r)
      d_L.release()
      mat.release()
    }}
  }

  // produces a stream of random square matrices
  def getMatrices(initSize: Int, step: Int = 32, count: Int = 20)(implicit handle: cublasHandle): Stream[CuMatrix[Double]] =
    Stream.iterate(initSize, count) { _ + step } map { i => CuMatrix.fromDense(DenseMatrix.rand[Double](i, i)) }


}
