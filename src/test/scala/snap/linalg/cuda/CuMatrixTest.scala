package snap.linalg.cuda

import org.scalatest.{BeforeAndAfterEach, FunSuite}
import jcuda.jcublas.{JCublas2, cublasHandle}
import breeze.linalg._
import jcuda.runtime.JCuda

/**
 * TODO
 *
 * @author dlwh
 **/
class CuMatrixTest extends org.scalatest.fixture.FunSuite {

  type FixtureParam = cublasHandle

  def withFixture(test: OneArgTest) {
    val handle = new cublasHandle()
    JCuda.setExceptionsEnabled(true)
    JCublas2.setExceptionsEnabled(true)
    JCublas2.cublasCreate(handle)

    try {
      withFixture(test.toNoArgTest(handle)) // "loan" the fixture to the test
    }
    finally JCublas2.cublasDestroy(handle)
  }



  test("fromDense and back") { (_handle: cublasHandle) =>
    implicit val handle = _handle
    val rand = convert(DenseMatrix.rand(10, 12), Float)
    val cumat = CuMatrix.zeros[Float](10, 12)
    cumat := rand
    val dense = cumat.toDense
    assert(dense === rand)
  }

  test("copy gpuside") { (_handle: cublasHandle) =>
    implicit val handle = _handle
    val rand = convert(DenseMatrix.rand(10, 12), Float)
    val cumat, cumat2 = CuMatrix.zeros[Float](10, 12)
    cumat := rand
    cumat2 := cumat
    val dense = cumat2.toDense
    assert(dense === rand)
  }

  test("fromDense transpose and back") { (_handle: cublasHandle) =>
    implicit val handle = _handle
    val rand = convert(DenseMatrix.rand(12, 10), Float)
    val cumat = CuMatrix.zeros[Float](10, 12)
    cumat := rand.t
    val dense = cumat.toDense
    assert(dense.rows === rand.cols)
    assert(dense.cols === rand.rows)
    assert(dense === rand.t)
  }

  test("fromDense transpose and back 2") { (_handle: cublasHandle) =>
    implicit val handle = _handle
    val rand = convert(DenseMatrix.rand(12, 10), Float)
    val cumat = CuMatrix.zeros[Float](10, 12)

    cumat.t := rand
    val dense2 = cumat.toDense
    assert(dense2.rows === rand.cols)
    assert(dense2.cols === rand.rows)
    assert(dense2 === rand.t)
  }

  test("copy transpose gpuside") { (_handle: cublasHandle) =>
    implicit val handle = _handle
    val rand = convert(DenseMatrix.rand(10, 12), Float)
    val cumat = CuMatrix.zeros[Float](10, 12)
    val cumat2 = CuMatrix.zeros[Float](12, 10)
    cumat := rand
    cumat2 := cumat.t
    val dense = cumat2.toDense
    assert(dense.t === rand)
  }

  test("copy transpose gpuside 2") { (_handle: cublasHandle) =>
    implicit val handle = _handle
    val rand = convert(DenseMatrix.rand(10, 12), Float)
    val cumat = CuMatrix.zeros[Float](10, 12)
    val cumat2 = CuMatrix.zeros[Float](12, 10)
    cumat := rand
    cumat2.t := cumat
    val dense = cumat2.toDense
    assert(dense.t === rand)
  }



}
