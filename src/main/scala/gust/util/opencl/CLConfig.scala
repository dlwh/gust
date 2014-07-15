package gust.util.opencl

import com.nativelibs4java.opencl.{JavaCL, CLQueue, CLContext}
import com.nativelibs4java.util.IOUtils

/**
 * Created by Piotr on 2014-07-14.
 */

object CLConfig {
  implicit val context: CLContext = JavaCL.createBestContext()
  implicit val queue: CLQueue = context.createDefaultQueue()
}
