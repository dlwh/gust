package gust.util.cuda

import jcuda.driver.{CUstream, CUfunction, CUcontext}
import jcuda.driver.JCudaDriver._

object CuKernel {
  def invoke(fn: CUfunction, gridDims: Dim3, blockDims: Dim3, sharedMemoryBytes: Int = 0)(args: Any*)(implicit context: CuContext):Unit = {
    context.withPush {
      val params = setupKernelParameters(args:_*)
      cuLaunchKernel(fn,
        gridDims.x, gridDims.y, gridDims.z,
        blockDims.x, blockDims.y, blockDims.z,
        sharedMemoryBytes, new CUstream(),
        params, null)
      jcuda.runtime.JCuda.cudaFreeHost(params)

    }
  }

  /**
   * from VecUtils
   * Create a pointer to the given arguments that can be used as
   * the parameters for a kernel launch.
   *
   * @param args The arguments
   * @return The pointer for the kernel arguments
   * @throws NullPointerException If one of the given arguments is
   *                              <code>null</code>
   * @throws CudaException If one of the given arguments has a type
   *                       that can not be passed to a kernel (that is, a type that is
   *                       neither primitive nor a { @link Pointer})
   */
  private def setupKernelParameters(args: Any*) = {
    import java.lang._
    import jcuda.Pointer
    val kernelParameters: Array[CuPointer] = new Array[CuPointer](args.length)
    for( (arg, i) <- args.zipWithIndex) {
      arg match {
        case null =>
          throw new NullPointerException("Argument " + i + " is null")
        case argPointer: Pointer =>
          val pointer: Pointer = Pointer.to(argPointer)
          kernelParameters(i) = pointer
        case value: Byte =>
          val pointer: Pointer = Pointer.to(Array[scala.Byte](value))
          kernelParameters(i) = pointer
        case value: Short =>
          val pointer: Pointer = Pointer.to(Array[scala.Short](value))
          kernelParameters(i) = pointer
        case value: Integer =>
          val pointer: Pointer = Pointer.to(Array[scala.Int](value))
          kernelParameters(i) = pointer
        case value: Long =>
          val pointer: Pointer = Pointer.to(Array[scala.Long](value))
          kernelParameters(i) = pointer
        case value: Float =>
          val pointer: Pointer = Pointer.to(Array[scala.Float](value))
          kernelParameters(i) = pointer
        case value: Double =>
          val pointer: Pointer = Pointer.to(Array[scala.Double](value))
          kernelParameters(i) = pointer
        case _ =>
          throw new RuntimeException("Type " + arg.getClass + " may not be passed to a function")
      }
    }
    jcuda.Pointer.to(kernelParameters:_*)
  }

}