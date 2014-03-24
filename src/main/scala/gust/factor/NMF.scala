package gust.factor

import breeze.linalg._
import gust.linalg.cuda.CuMatrix
import breeze.numerics.abs

/**
 * TODO
 *
 * @author dlwh
 **/
object NMF {

  def supervised(W: CuMatrix[Float], X: CuMatrix[Float], iters: Int = 200, eps: Float = 1E-6f) = {
    require(W.rows == X.rows)
    import W.blas
    val n = X.rows
    val m = X.cols
    val r = W.cols

    var H = CuMatrix.ones[Float](r, m)
    val Wones = W.t * CuMatrix.ones[Float](n, 1) * CuMatrix.ones[Float](1, m)

    for(i <- 0 until iters) {
      println(i)
      H = H :*= (W.t * (X :/ (W * H))) :/= Wones
      max.inPlace(H, eps)

      System.gc()
    }

    H
  }

  def main(args: Array[String]) {
    import jcuda.jcublas._
    implicit  val handle = new cublasHandle
    JCublas2.cublasCreate(handle)
    val in = System.currentTimeMillis()
    val W = CuMatrix.ones[Float](1024, 108)
    val H = CuMatrix.ones[Float](108, 10000)
    val X = W * H
    val Hhat = supervised(W, X)
    val Xhat = W * Hhat
    val out = System.currentTimeMillis()
    println(max(abs(Xhat.toDense - X.toDense)))
    println(s"${(out - in)/1000.0} seconds")
  }

}
