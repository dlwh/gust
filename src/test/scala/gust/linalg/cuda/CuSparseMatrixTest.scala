package gust.linalg.cuda

import jcuda.jcusparse.{JCusparse2, cusparseHandle}
import org.scalatest.{BeforeAndAfterEach, FunSuite}
import jcuda.jcublas.{JCublas2, cublasHandle}
import breeze.linalg._
import jcuda.runtime.JCuda
import breeze.numerics.{abs, cos}
import jcuda.driver.JCudaDriver
import gust.util.cuda.CuContext


/**
 * Created by Piotr on 2014-07-30.
 */
class CuSparseMatrixTest extends org.scalatest.fixture.FunSuite {

  type FixtureParam = (cublasHandle, cusparseHandle)

  def withFixture(test: OneArgTest) {
    val handle = new cublasHandle()
    JCuda.setExceptionsEnabled(true)
    JCublas2.setExceptionsEnabled(true)
    JCublas2.cublasCreate(handle)

    val sp_handle = new cusparseHandle
    JCusparse2.cusparseCreate(sp_handle)

    val fixt = (handle, sp_handle)

    try {
      withFixture(test.toNoArgTest(fixt)) // "loan" the fixture to the test
    }
    finally {
      JCublas2.cublasDestroy(handle)
      JCusparse2.cusparseDestroy(sp_handle)
    }
  }


  test("fromDense and back") { (fixt: (cublasHandle, cusparseHandle)) =>
    implicit val handle = fixt._1
    implicit val sp_handle = fixt._2

    // this matrix will not be sparse but that makes little difference
    val rand = convert(DenseMatrix.rand(10, 12), Float)
    val cuspmat = CuSparseMatrix.fromDense(rand)

    val dense = cuspmat.toDense
    assert(dense === rand)
  }

  test("fromCuMatrix and back") { (fixt: (cublasHandle, cusparseHandle)) =>
    implicit val handle = fixt._1
    implicit val sp_handle = fixt._2

    // this matrix will not be sparse but that makes little difference
    val rand = convert(DenseMatrix.rand(10, 12), Float)
    val cumat = CuMatrix.fromDense(rand)
    val cuspmat = CuSparseMatrix.fromCuMatrix(cumat)

    val cumat2 = cuspmat.toCuMatrix
    val dense = cumat2.toDense
    assert(dense === rand)
  }

  test("transpose") { (fixt: (cublasHandle, cusparseHandle)) =>
    implicit val handle = fixt._1
    implicit val sp_handle = fixt._2

    // this matrix will not be sparse but that makes little difference
    val rand: DenseMatrix[Float] = convert(DenseMatrix.rand(10, 12), Float)
    val cuspmat = CuSparseMatrix.fromDense(rand).transpose

    val dense = cuspmat.toDense
    assert(dense === rand.t)
  }



}