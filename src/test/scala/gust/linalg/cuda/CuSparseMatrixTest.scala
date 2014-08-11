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

  test("sparse addition") { (fixt: (cublasHandle, cusparseHandle)) =>
    implicit val handle = fixt._1
    implicit val sp_handle = fixt._2

    val dense1: DenseMatrix[Double] = DenseMatrix((1.0, 4.0, 0.0, 0.0, 0.0), (0.0, 2.0, 3.0, 0.0, 0.0), (5.0, 0.0, 0.0, 7.0, 8.0), (0.0, 0.0, 9.0, 0.0, 6.0))
    val dense2: DenseMatrix[Double] = DenseMatrix((0.0, 14.0, 0.0, 2.0, 0.0), (0.0, 0.0, 0.0, 5.0, 6.0), (0.0, 10.0, 0.0, 0.0, 1.0), (7.0, 0.0, 2.0, 3.0, 0.0))
    val cuspmat1 = CuSparseMatrix.fromDense(dense1)
    val cuspmat2 = CuSparseMatrix.fromDense(dense2)

    assert(dense1 + dense2 === (cuspmat1 + cuspmat2).toDense)
  }

  test("sparse addition with transpose") { (fixt: (cublasHandle, cusparseHandle)) =>
    implicit val handle = fixt._1
    implicit val sp_handle = fixt._2

    val dense1: DenseMatrix[Double] = DenseMatrix((1.0, 4.0, 0.0, 0.0, 0.0), (0.0, 2.0, 3.0, 0.0, 0.0), (5.0, 0.0, 0.0, 7.0, 8.0), (0.0, 0.0, 9.0, 0.0, 6.0))
    val dense2: DenseMatrix[Double] = DenseMatrix((0.0, 14.0, 0.0, 2.0, 0.0), (0.0, 0.0, 0.0, 5.0, 6.0), (0.0, 10.0, 0.0, 0.0, 1.0), (7.0, 0.0, 2.0, 3.0, 0.0))
    val cuspmat1 = CuSparseMatrix.fromDense(dense1)
    val cuspmat2 = CuSparseMatrix.fromDense(dense2)

    assert(dense1.t + dense2.t === (cuspmat1.t + cuspmat2.t).toDense)
  }

  test("sparse subtraction") { (fixt: (cublasHandle, cusparseHandle)) =>
    implicit val handle = fixt._1
    implicit val sp_handle = fixt._2

    val dense1: DenseMatrix[Double] = DenseMatrix((1.0, 4.0, 0.0, 0.0, 0.0), (0.0, 2.0, 3.0, 0.0, 0.0), (5.0, 0.0, 0.0, 7.0, 8.0), (0.0, 0.0, 9.0, 0.0, 6.0))
    val dense2: DenseMatrix[Double] = DenseMatrix((0.0, 14.0, 0.0, 2.0, 0.0), (0.0, 0.0, 0.0, 5.0, 6.0), (0.0, 10.0, 0.0, 0.0, 1.0), (7.0, 0.0, 2.0, 3.0, 0.0))
    val cuspmat1 = CuSparseMatrix.fromDense(dense1)
    val cuspmat2 = CuSparseMatrix.fromDense(dense2)

    assert(dense1 - dense2 === (cuspmat1 - cuspmat2).toDense)
  }

  test("sparse multiplication") { (fixt: (cublasHandle, cusparseHandle)) =>
    implicit val handle = fixt._1
    implicit val sp_handle = fixt._2

    val dense1: DenseMatrix[Double] = DenseMatrix((1.0, 4.0, 0.0, 0.0, 0.0), (0.0, 2.0, 3.0, 0.0, 0.0), (5.0, 0.0, 0.0, 7.0, 8.0), (0.0, 0.0, 9.0, 0.0, 6.0))
    val dense2: DenseMatrix[Double] = DenseMatrix((0.0, 14.0, 0.0, 2.0), (0.0, 0.0, 1.0, 0.0), (5.0, 6.0, 0.0, 10.0), (0.0, 0.0, 1.0, 7.0), (0.0, 2.0, 3.0, 0.0))
    val cuspmat1 = CuSparseMatrix.fromDense(dense1)
    val cuspmat2 = CuSparseMatrix.fromDense(dense2)

    val denseProd: DenseMatrix[Double] = dense1 * dense2
    val sparseProd: CuSparseMatrix[Double] = cuspmat1 * cuspmat2

    assert(denseProd === sparseProd.toDense)
  }

  test("sparse multiplication with transpose") { (fixt: (cublasHandle, cusparseHandle)) =>
    implicit val handle = fixt._1
    implicit val sp_handle = fixt._2

    val dense1: DenseMatrix[Double] = DenseMatrix((1.0, 4.0, 0.0, 0.0, 0.0), (0.0, 2.0, 3.0, 0.0, 0.0), (5.0, 0.0, 0.0, 7.0, 8.0), (0.0, 0.0, 9.0, 0.0, 6.0))
    val dense2: DenseMatrix[Double] = DenseMatrix((0.0, 14.0, 0.0, 2.0, 0.0), (0.0, 1.0, 0.0, 5.0, 6.0), (0.0, 10.0, 0.0, 0.0, 1.0), (7.0, 0.0, 2.0, 3.0, 0.0))
    val cuspmat1 = CuSparseMatrix.fromDense(dense1)
    val cuspmat2 = CuSparseMatrix.fromDense(dense2)

    val denseProd: DenseMatrix[Double] = dense1 * dense2.t
    val sparseProd: CuSparseMatrix[Double] = cuspmat1 * cuspmat2.t

    assert(denseProd === sparseProd.toDense)
  }

  test("sparse matrix * dense vector") { (fixt: (cublasHandle, cusparseHandle)) =>
    implicit val handle = fixt._1
    implicit val sp_handle = fixt._2

    val dense: DenseMatrix[Double] = DenseMatrix((1.0, 4.0, 0.0, 0.0, 0.0), (0.0, 2.0, 3.0, 0.0, 0.0), (5.0, 0.0, 0.0, 7.0, 8.0), (0.0, 0.0, 9.0, 0.0, 6.0))
    val densevec: DenseMatrix[Double] = DenseMatrix((1.0), (2.0), (3.0), (4.0), (5.0))
    val cuspmat = CuSparseMatrix.fromDense(dense)
    val cuvec = CuMatrix.fromDense(densevec)

    val denseProd: DenseMatrix[Double] = dense * densevec
    val sparseProd: CuMatrix[Double] = cuspmat * cuvec

    assert(denseProd === sparseProd.toDense)
  }

  test("sparse matrix * dense vector with transpose") { (fixt: (cublasHandle, cusparseHandle)) =>
    implicit val handle = fixt._1
    implicit val sp_handle = fixt._2

    val dense: DenseMatrix[Double] = DenseMatrix((1.0, 4.0, 0.0, 0.0, 0.0), (0.0, 2.0, 3.0, 0.0, 0.0), (5.0, 0.0, 0.0, 7.0, 8.0), (0.0, 0.0, 9.0, 0.0, 6.0))
    val densevec: DenseMatrix[Double] = DenseMatrix((1.0), (2.0), (3.0), (4.0))
    val cuspmat = CuSparseMatrix.fromDense(dense)
    val cuvec = CuMatrix.fromDense(densevec)

    val denseProd: DenseMatrix[Double] = dense.t * densevec
    val sparseProd: CuMatrix[Double] = cuspmat.t * cuvec

    assert(denseProd === sparseProd.toDense)
  }

  test("sparse matrix * scalar") { (fixt: (cublasHandle, cusparseHandle)) =>
    implicit val handle = fixt._1
    implicit val sp_handle = fixt._2

    val cuspmat = CuSparseMatrix.fromDense(DenseMatrix((1.0, 4.0, 0.0, 0.0, 0.0), (0.0, 2.0, 3.0, 0.0, 0.0), (5.0, 0.0, 0.0, 7.0, 8.0), (0.0, 0.0, 9.0, 0.0, 6.0)))
    val sparseProd: CuSparseMatrix[Double] = cuspmat * 2.0
    val dense = DenseMatrix((2.0, 8.0, 0.0, 0.0, 0.0), (0.0, 4.0, 6.0, 0.0, 0.0), (10.0, 0.0, 0.0, 14.0, 16.0), (0.0, 0.0, 18.0, 0.0, 12.0))

    assert(dense === sparseProd.toDense)
  }

}