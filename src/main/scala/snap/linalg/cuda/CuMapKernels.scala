package snap.linalg.cuda

import breeze.generic.UFunc
import snap.util.cuda._
import jcuda.Pointer
import scala.reflect.ClassTag

/**
 * TODO
 *
 * @author dlwh
 **/
class CuMapKernels[X, T:ClassTag](typeName: String) {

  private val module:CuModule = {
    CuModule(getClass.getResourceAsStream(s"map_kernels_$typeName.ptx"))
  }

  def implFor[K<:UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext):UFunc.UImpl[K, CuMatrix[T], CuMatrix[T]] = {
    val kern = module.getKernel6[Int, Int, Pointer, Int, Pointer, Int](s"map_${funName}_$typeName")


    new UFunc.UImpl[K, CuMatrix[T], CuMatrix[T]] {
      def apply(v: CuMatrix[T]): CuMatrix[T] = {
        import v.blas
        val res = CuMatrix.create[T](v.rows, v.cols)
        kern(512, 20, 1)(v.rows, v.cols, res.data.toCuPointer, res.majorStride, v.data.toCuPointer, res.majorStride)
        res
      }
    }
  }
}
