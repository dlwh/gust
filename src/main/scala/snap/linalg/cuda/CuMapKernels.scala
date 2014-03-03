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
        val res = if(v.isTranspose) CuMatrix.create[T](v.cols, v.rows).t else  CuMatrix.create[T](v.rows, v.cols)
        val minorSize = if(v.isTranspose) v.cols else v.rows
        kern(512, 20, 1)(minorSize, v.majorSize, res.data.toCuPointer, res.majorStride, v.data.toCuPointer, res.majorStride)
        res
      }
    }
  }
}
