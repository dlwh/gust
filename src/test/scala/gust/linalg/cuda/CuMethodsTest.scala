package gust.linalg.cuda

import org.scalameter.api._

import jcuda.jcublas.{cublasOperation, cublasHandle, JCublas2, cublasDiagType, cublasFillMode}
import gust.util.cuda
import jcuda.runtime.{cudaError, cudaMemcpyKind, cudaStream_t, JCuda}
import cuda._
import breeze.linalg.{DenseVector, DenseMatrix}

/**
 * Created by piotrek on 31.05.2014.
 */
//class CuMethodsTest {
//
//}

object CuMethodsTest
  extends PerformanceTest.Quickbenchmark {

  val sizes = Gen.range("size")(256, 2048, 256)
  implicit val handle = new cublasHandle
  JCublas2.cublasCreate(handle)

  val h_Sets: Gen[(DenseMatrix[Double], DenseVector[Double])] = sizes map {
    size => (DenseMatrix.rand[Double](size, size), DenseVector.rand[Double](size))
  }

  val d_Sets: Gen[(CuMatrix[Double], CuMatrix[Double])] = h_Sets map {
    hset => ( CuMatrix.fromDense(hset._1),
              CuMatrix.fromDense(hset._2.asDenseMatrix.t) )
  }


  performance of "CuMethods" in {
    measure method "solve2" in {
      using(d_Sets) in {
        set => {
          set._1 \ set._2
        }
      }
    }
  }

  performance of "Breeze.linalg" in {
    measure method "solve" in {
      using(h_Sets) in {
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