package gust.util.opencl

import com.nativelibs4java.opencl.CLKernel
import com.nativelibs4java.util.IOUtils
import gust.util.opencl.CLConfig._

/**
 * Created by Piotr on 2014-07-14.
 */
object CLKernelManager {
  var kernels = Map.empty[String, CLKernel]

  def prepareKernel(name: String) = {
    val src = IOUtils.readText(this.getClass.getResource("/gust/linalg/opencl/cl_kernels_float.cl"))
    val program = context.createProgram(src)
    val kernel = program.createKernel(name)

    kernel
  }

  def getKernel(name: String) = {
    if (kernels.contains(name)) kernels(name)
    else {
      val kernel = prepareKernel(name)
      kernels += (name -> kernel)

      kernel
    }
  }
}
