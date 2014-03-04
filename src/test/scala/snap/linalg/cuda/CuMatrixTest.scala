package snap.linalg.cuda

import org.scalatest.{BeforeAndAfterEach, FunSuite}
import jcuda.jcublas.{JCublas2, cublasHandle}
import breeze.linalg._
import jcuda.runtime.JCuda
import breeze.numerics.{abs, cos}
import jcuda.driver.JCudaDriver
import snap.util.cuda.CuContext

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

  test("rand test") { (_handle: cublasHandle) =>
    implicit val handle = _handle
    val cumat = CuMatrix.rand(10, 12)
    val dense = cumat.toDense
    assert(all(dense))
    assert(dense.forallValues(_ < 1))
  }

  test("Multiply") { (_handle: cublasHandle) =>
    implicit val handle = _handle
    val a = DenseMatrix((1.0f, 2.0f, 3.0f),(4.0f, 5.0f, 6.0f))
    val b = DenseMatrix((7.0f, -2.0f, 8.0f),(-3.0f, -3.0f, 1.0f),(12.0f, 0.0f, 5.0f))
    val ga = CuMatrix.fromDense(a)
    val gb = CuMatrix.fromDense(b)

    assert( (ga * gb).toDense === DenseMatrix((37.0f, -8.0f, 25.0f), (85.0f, -23.0f, 67.0f)))


    val x = ga * ga.t
    assert(x.toDense === DenseMatrix((14.0f,32.0f),(32.0f,77.0f)))

    val y = ga.t * ga
    assert(y.toDense === DenseMatrix((17.0f,22.0f,27.0f),(22.0f,29.0f,36.0f),(27.0f,36.0f,45.0f)))

//    val z  = gb * (gb + 1.0f)
//    assert(z.toDense === DenseMatrix((164.0f,5.0f,107.0f),(-5.0f,10.0f,-27.0f),(161.0f,-7.0f,138.0f)))
  }

  test("Reshape") { (_handle: cublasHandle) =>
    implicit val handle = _handle
    val dm = convert(DenseMatrix.rand(20, 30), Float)
    val cu = CuMatrix.zeros[Float](20, 30)
    cu := dm
    assert(cu.reshape(10, 60).toDense ===  dm.reshape(10, 60))
  }

  test("Slices") { (_handle: cublasHandle) =>
    implicit val handle = _handle
    val dm : DenseMatrix[Float] = convert(DenseMatrix.rand(20, 30), Float)
    val cu = CuMatrix.zeros[Float](20, 30)
    cu := dm

    assert(cu(0, ::).toDense === dm(0, ::))
    assert(cu(0, 1 to 4).toDense === dm(0, 1 to 4), s"Full matrix: $dm")
    assert(cu(::, 0).toDense === dm(::, 0).toDenseMatrix.t, s"${dm(::, 0)}")
    assert(cu(1 to 4, 0).toDense === dm(1 to 4, 0).toDenseMatrix.t, s"Full matrix: $dm")
    assert(cu.t(0, ::).toDense === dm.t(0, ::))
    assert(cu.t(0, 1 to 4).toDense === dm.t(0, 1 to 4), s"Full matrix: $dm")
    assert(cu.t(::, 0).toDense === dm.t(::, 0).toDenseMatrix.t, s"${dm(::, 0)}")
    assert(cu.t(1 to 4, 0).toDense === dm.t(1 to 4, 0).toDenseMatrix.t, s"Full matrix: $dm")

  }

  test("Basic mapping functions") { (_handle: cublasHandle) =>
    implicit val handle = _handle
    val dm : DenseMatrix[Float] = convert(DenseMatrix.rand(30, 10), Float)
    val cosdm = cos(dm)
    val cu = CuMatrix.zeros[Float](30, 10)
    cu := dm
    assert(cu.toDense === dm)
//    import CuMatrix.kernelsFloat
    val coscu = cos(cu)
    assert( max(abs(cosdm - coscu.toDense)) < 1E-5, s"$cosdm ${coscu.toDense}")



  }


  test("Basic mapping functions transpose") { (_handle: cublasHandle) =>
    implicit val handle = _handle
    val dm : DenseMatrix[Float] = convert(DenseMatrix.rand(30, 10), Float)
    val cosdm = cos(dm)
    val cu = CuMatrix.zeros[Float](30, 10)
    cu := dm
    assert(cu.toDense === dm)
//    import CuMatrix.kernelsFloat
    val coscu = cos(cu.t)
    assert( max(abs(cosdm.t - coscu.toDense)) < 1E-5, s"$cosdm ${coscu.toDense}")



  }

  test("Basic ops functions") { (_handle: cublasHandle) =>
    implicit val handle = _handle
    val dm : DenseMatrix[Float] = convert(DenseMatrix.rand(30, 10), Float)
    val cu = CuMatrix.zeros[Float](30, 10)
    cu := dm
    assert(cu.toDense === dm)
//    import CuMatrix.kernelsFloat
    val cu2 = cu + cu
    assert( max(abs((dm * 2.0f) - cu2.toDense)) < 1E-5)
    assert( max(abs((dm * 2.0f) - (cu * 2.0f).toDense)) < 1E-5)
  }

  test("broadcast addition") { (_handle: cublasHandle) =>
    implicit val handle = _handle
    val dm : DenseMatrix[Float] = convert(DenseMatrix.rand(30, 10), Float)
    val cu = CuMatrix.zeros[Float](30, 10)
    cu := dm

    val dmadd = dm(::, *) + dm(::, 1)
    val cuadd = cu(::, *) + cu(::, 1)

    assert(cuadd.toDense === dmadd)

//    val dmadd2 =  dm(::, 1) + dm(::, *)
    val cuadd2 =  cu(::, 1) + cu(::, *)

    assert(cuadd2.toDense === dmadd)
  }
}
