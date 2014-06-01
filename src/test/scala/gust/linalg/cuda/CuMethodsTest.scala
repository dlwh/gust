package gust.linalg.cuda

import org.scalameter.api._

import jcuda.jcublas.{cublasHandle, JCublas2}
import gust.util.cuda
import breeze.linalg.{DenseVector, DenseMatrix}

/**
 * Created by piotrek on 31.05.2014.
 */
//class CuMethodsTest {
//
//}

object CuMethodsTest
  extends PerformanceTest.Quickbenchmark {

  // this has to do for generating float DenseMatrices/Vectors for now
  def randFloatMatrix(rows: Int, cols: Int) = DenseMatrix.tabulate(rows, cols) { (i, j) => Math.random().toFloat }
  def randFloatVector(size: Int) = DenseVector.tabulate(size) { i => Math.random().toFloat }

  val sizes = Gen.range("size")(256, 2048, 256)
  implicit val handle = new cublasHandle
  JCublas2.cublasCreate(handle)

  val h_SetsDouble: Gen[(DenseMatrix[Double], DenseVector[Double])] = sizes map {
    size => (DenseMatrix.rand[Double](size, size), DenseVector.rand[Double](size))
  }

  val h_SetsFloat: Gen[(DenseMatrix[Float], DenseVector[Float])] = sizes map {
    size => (randFloatMatrix(size, size), randFloatVector(size))
  }

  val d_SetsDouble: Gen[(CuMatrix[Double], CuMatrix[Double])] = h_SetsDouble map {
    hset => ( CuMatrix.fromDense(hset._1),
              CuMatrix.fromDense(hset._2.asDenseMatrix.t) )
  }

  val d_SetsFloat: Gen[(CuMatrix[Float], CuMatrix[Float])] = h_SetsFloat map {
    hset => ( CuMatrix.fromDense(hset._1),
      CuMatrix.fromDense(hset._2.asDenseMatrix.t) )
  }


  performance of "CuMethods" in {
    measure method "solveDouble" in {
      using(d_SetsDouble) in {
        set => {
          set._1 \ set._2
        }
      }
    }

    measure method "solveFloat" in {
      using(d_SetsFloat) in {
        set => {
          set._1 \ set._2
        }
      }
    }
  }

  performance of "Breeze.linalg" in {
    measure method "solve (Double)" in {
      using(h_SetsDouble) in {
        set => {
          set._1 \ set._2
        }
      }
    }

    measure method "solve (Float)" in {
      using(h_SetsFloat) in {
        set => {
          set._1 \ set._2
        }
      }
    }
  }

//  performance of "CuMethods" in {
//    measure method "solveSlow" in {
//      using(d_Sets) in {
//        set => {
//          CuMethods.solveSlow(set._1, set._2)
//        }
//      }
//    }
//  }
}