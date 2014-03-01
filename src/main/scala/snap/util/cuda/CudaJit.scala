package snap.util.cuda

import scala.io.Source
import jcuda.driver._
import jcuda.driver.JCudaDriver._
import jcuda.Pointer

/**
 * TODO
 *
 * @author dlwh
 **/
object CudaJit {

  def mappingFunctionText(name: String, op: (String, String, Int)=>String) = {
    val resource = Source.fromInputStream(this.getClass.getResourceAsStream("map_kernel.ptx"))
    val str = resource.mkString
    resource.close()

    str.replaceAll("[$][{]name}", name).replaceAll("[$][{]operation}", op("%f1", "%f4", 5))
  }



  def mapKernelFromPtx(name: String, op: (String, String, Int)=>String)(implicit context: CUcontext) = {
    val text = mappingFunctionText(name, op)

    val module = contextLock.synchronized {
      cuInit(0)
      // Try to obtain the current context
      cuCtxSetCurrent(context)

      val module = new CUmodule()

      val ptxData = (text).getBytes("UTF-8")

      cuModuleLoadDataEx(module, Pointer.to(ptxData),
        0, Array(), Pointer.to(Array[Int]()))
      module
    }


    val function = new CUfunction()
    cuModuleGetFunction(function, module, name)

    new CuKernel3[CuPointer, Int, CuPointer](function, Array(getMaxBlockDimX(0)))
  }

  /**
   *
   * from VecUtils in JCUDA
   * Obtain the CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X of the current device
   *
   * @return The maximum block dimension, in x-direction
   */
  private def getMaxBlockDimX(deviceNumber: Int) = {
    val device = new CUdevice()
    cuDeviceGet(device, deviceNumber)
    val maxBlockDimX = Array(0)
    cuDeviceGetAttribute(maxBlockDimX,
      CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device)
    maxBlockDimX.head
  }

}
