package snap.util.cuda

import org.scalatest.FunSuite
import jcuda.driver.{JCudaDriver, CUdevice, CUcontext}
import jcuda.driver.JCudaDriver._
import snap.linalg.cuda.CuMatrix
import jcuda.jcublas.{JCublas2, cublasHandle}
import jcuda.runtime.JCuda
import breeze.linalg.DenseMatrix

/**
 * TODO
 *
 * @author dlwh
 **/
class CudaJitTest extends FunSuite {

  test("cos test") {
    cuInit(0)
    implicit val ctx = createContext()
    implicit val handle = new cublasHandle()
    JCudaDriver.setExceptionsEnabled(true)
    JCuda.setExceptionsEnabled(true)
    JCublas2.setExceptionsEnabled(true)
    JCublas2.cublasCreate(handle)

    val kernel = CudaJit.mapKernelFromPtx("cos", (in, out, _) => s"cos.approx.f32 $out, $in;")
    val mat = CuMatrix.zeros[Float](10, 12)
    kernel(10 * 12)(mat.data.toCuPointer, 120, mat.data.toCuPointer)
    val res = mat.toDense
    assert(res === DenseMatrix.ones[Float](10, 12))
  }

  private def createContext() = {


    val device = new CUdevice();
    cuDeviceGet(device, 0)
    val context = new CUcontext()
    cuCtxCreate(context, 0, device)
    context
  }

}
