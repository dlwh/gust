package gust.factor

import breeze.linalg._
import gust.linalg.cuda.CuMatrix
import breeze.numerics.abs
import jcuda.jcublas.cublasHandle
import breeze.stats.distributions.Rand

/**
 * TODO
 *
 * @author dlwh
 **/
object NMF {

  def euclidean(X: CuMatrix[Float], dims: Int, iters: Int = 200, eps: Float = 1E-6f)(implicit blas: cublasHandle) = {
    val n = X.rows
    val m = X.cols
    val r = dims
    val W = CuMatrix.rand(n, r)
    val H = CuMatrix.rand(r, m)

    for(i <- 0 until iters) {
      println(i)
      H :*= ((W.t * X) :/= (W.t * W * H))
      W :*= ((X * H.t) :/= (W * H * H.t))
//      System.gc()
    }

    (W, H)

  }

  def euclideanBreeze(X: DenseMatrix[Float], dims: Int, iters: Int = 200, eps: Float = 1E-6f) = {
    val n = X.rows
    val m = X.cols
    val r = dims
    val W = DenseMatrix.rand(n, r, Rand.uniform.map(_.toFloat))
    val H = DenseMatrix.rand(r, m, Rand.uniform.map(_.toFloat))

    for(i <- 0 until iters) {
      println(i)
      H :*= ((W.t * X) :/= (W.t * W * H))
      W :*= ((X * H.t) :/= (W * H * H.t))
      System.gc()
    }

    (W, H)

  }

  def testGPU(X: DenseMatrix[Float], hidden: Int)(implicit blas: cublasHandle) = {
    val (w, h) = euclidean(CuMatrix.fromDense(X), hidden)
    val xhat = (w * h).toDense
    max(abs(xhat - X))
  }

  def testCPU(X: DenseMatrix[Float], hidden: Int) = {
    val (w, h) = euclideanBreeze(X, hidden)
    val xhat = (w * h)
    max(abs(xhat - X))
  }

  def main(args: Array[String]) {
    import jcuda.jcublas._
    implicit  val handle = new cublasHandle
    JCublas2.cublasCreate(handle)
    val hidden = 8
    val W = convert(DenseMatrix.rand(2000, hidden), Float)
    val H = convert(DenseMatrix.rand(hidden, 2500), Float)
    val X = W * H
    val in = System.currentTimeMillis()
    val error = testGPU(X, hidden)
    val out = System.currentTimeMillis()
    val error2 = testCPU(X, hidden)
    val out2 = System.currentTimeMillis()
    println(s"gpu $error ${(out - in)/1000.0} seconds")
    println(s"cpu $error2 ${(out2 - out)/1000.0} seconds")
  }

}
