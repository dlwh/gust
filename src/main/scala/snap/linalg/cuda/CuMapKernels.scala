package snap.linalg.cuda

import breeze.generic.UFunc
import snap.util.cuda._
import jcuda.Pointer
import scala.reflect.ClassTag
import java.util.concurrent.ConcurrentHashMap

/**
 * TODO
 *
 * @author dlwh
 **/
class CuMapKernels[X, T:ClassTag](typeName: String) {

  private val module:CuModule = {
    CuModule(getClass.getResourceAsStream(s"map_kernels_$typeName.ptx"))
  }

  private val implCache = new ConcurrentHashMap[String, CuKernel6[Int, Int, Pointer, Int, Pointer, Int]]
  private val impl2Cache = new ConcurrentHashMap[String, CuKernel8[Int, Int, Pointer, Int, Pointer, Int, Pointer, Int]]
  private val impl2VSCache = new ConcurrentHashMap[String, CuKernel7[Int, Int, Pointer, Int, Pointer, Int, T]]
  private val impl2SVCache = new ConcurrentHashMap[String, CuKernel7[Int, Int, Pointer, Int, T, Pointer, Int]]

  def implFor[K<:UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext):UFunc.UImpl[K, CuMatrix[T], CuMatrix[T]] = {
    var kern = implCache.get(funName)
    if(kern == null) {
      kern = module.getKernel6[Int, Int, Pointer, Int, Pointer, Int](s"map_${funName}_$typeName")
      implCache.put(funName, kern)
    }


    new UFunc.UImpl[K, CuMatrix[T], CuMatrix[T]] {
      def apply(v: CuMatrix[T]): CuMatrix[T] = {
        import v.blas
        val res = if(v.isTranspose) CuMatrix.create[T](v.cols, v.rows).t else  CuMatrix.create[T](v.rows, v.cols)
        val minorSize = if(v.isTranspose) v.cols else v.rows
        kern(512, 20, 1)(minorSize, v.majorSize, res.data.toCuPointer, res.majorStride, v.data.toCuPointer, v.majorStride)
        res
      }
    }
  }

  def impl2For[K<:UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext):UFunc.UImpl2[K, CuMatrix[T], CuMatrix[T], CuMatrix[T]] = {
    var kern = impl2Cache.get(funName)
    if(kern == null) {
      kern = module.getKernel8[Int, Int, Pointer, Int, Pointer, Int, Pointer, Int](s"map2_${funName}_$typeName")
      impl2Cache.put(funName, kern)
    }


    new UFunc.UImpl2[K, CuMatrix[T], CuMatrix[T], CuMatrix[T]] {
      def apply(v: CuMatrix[T], v2: CuMatrix[T]): CuMatrix[T] = {
        require(v.rows == v2.rows && v.cols == v2.cols, "Dimension mismatch!")
        if(v.isTranspose != v2.isTranspose)
          throw new UnsupportedOperationException("Can't handle mixed transpose yet!")
        import v.blas
        val res = if(v.isTranspose) CuMatrix.create[T](v.cols, v.rows).t else  CuMatrix.create[T](v.rows, v.cols)
        val minorSize = if(v.isTranspose) v.cols else v.rows
        kern(512, 20, 1)(minorSize, v.majorSize, res.data.toCuPointer, res.majorStride, v.data.toCuPointer, v.majorStride, v2.data.toCuPointer, v2.majorStride)
        res
      }
    }
  }

  def impl2For_v_s[K<:UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext):UFunc.UImpl2[K, CuMatrix[T], T, CuMatrix[T]] = {
    var kern = impl2VSCache.get(funName)
    if(kern == null) {
      kern = module.getKernel7[Int, Int, Pointer, Int, Pointer, Int, T](s"map2_v_s_${funName}_$typeName")
      impl2VSCache.put(funName, kern)
    }


    new UFunc.UImpl2[K, CuMatrix[T], T, CuMatrix[T]] {
      def apply(v: CuMatrix[T], v2: T): CuMatrix[T] = {
        import v.blas
        val res = if(v.isTranspose) CuMatrix.create[T](v.cols, v.rows).t else  CuMatrix.create[T](v.rows, v.cols)
        val minorSize = if(v.isTranspose) v.cols else v.rows
        kern(512, 20, 1)(minorSize, v.majorSize, res.data.toCuPointer, res.majorStride, v.data.toCuPointer, v.majorStride, v2)
        res
      }
    }
  }

  def impl2For_s_v[K<:UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext):UFunc.UImpl2[K, T, CuMatrix[T], CuMatrix[T]] = {
    var kern = impl2SVCache.get(funName)
    if(kern == null) {
      kern = module.getKernel7[Int, Int, Pointer, Int, T, Pointer, Int](s"map2_s_v_${funName}_$typeName")
      impl2SVCache.put(funName, kern)
    }


    new UFunc.UImpl2[K, T, CuMatrix[T], CuMatrix[T]] {
      def apply(v2: T, v: CuMatrix[T]): CuMatrix[T] = {
        import v.blas
        val res = if(v.isTranspose) CuMatrix.create[T](v.cols, v.rows).t else  CuMatrix.create[T](v.rows, v.cols)
        val minorSize = if(v.isTranspose) v.cols else v.rows
        kern(512, 20, 1)(minorSize, v.majorSize, res.data.toCuPointer, res.majorStride, v2, v.data.toCuPointer, v.majorStride)
        res
      }
    }
  }
}
