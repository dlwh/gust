package gust.linalg.cuda

import breeze.generic.UFunc
import gust.util.cuda._
import jcuda.Pointer
import scala.reflect.ClassTag
import java.util.concurrent.ConcurrentHashMap
import breeze.linalg.{BroadcastedRows, BroadcastedColumns}

/**
 * TODO
 *
 * @author dlwh
 **/
trait CuVectorKernels { this: CuVector.type =>

  class KernelBroker[T: ClassTag] (typeName: String) {

    private val module: CuModule = {
      CuModule(getClass.getResourceAsStream(s"vector_kernels_$typeName.ptx"))
    }

    private val implCache = new ConcurrentHashMap[String, CuKernel5[Int, Pointer, Int, Pointer, Int]]
    private val impl2Cache = new ConcurrentHashMap[String, CuKernel7[Int, Pointer, Int, Pointer, Int, Pointer, Int]]
    private val impl2VSCache = new ConcurrentHashMap[String, CuKernel6[Int, Pointer, Int, Pointer, Int, T]]
    private val impl2SVCache = new ConcurrentHashMap[String, CuKernel6[Int, Pointer, Int, T, Pointer, Int]]
    private val reduceCache = new ConcurrentHashMap[String, CuKernel4[Int, Pointer, Pointer, Int]]
    private val colReduceCache = new ConcurrentHashMap[String, CuKernel4[Int, Pointer, Pointer, Int]]
    private val rowReduceCache = new ConcurrentHashMap[String, CuKernel4[Int, Pointer, Pointer, Int]]

    def implFor[K <: UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext): UFunc.UImpl[K, CuVector[T], CuVector[T]] = {
      var kern = implCache.get(funName)
      if (kern == null) {
        kern = module.getKernel5[Int, Pointer, Int, Pointer, Int](s"map_${funName}_$typeName")
        implCache.put(funName, kern)
      }


      new UFunc.UImpl[K, CuVector[T], CuVector[T]] {
        def apply(v: CuVector[T]): CuVector[T] = {
          val res = CuVector.create[T](v.length)
          kern((512, 1), (32, 1, 1))(v.length, res.offsetPointer, res.stride, v.offsetPointer, v.stride)
          res
        }
      }
    }

    def reducerFor[K<:UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext):UFunc.UImpl[K, CuVector[T], T] = {
      var kern = reduceCache.get(funName)
      if(kern == null) {
        kern = module.getKernel4[Int, Pointer, Pointer, Int](s"reduce_${funName}_$typeName")
        reduceCache.put(funName, kern)
      }

      val byteSize = org.bridj.BridJ.sizeOf(implicitly[ClassTag[T]].runtimeClass)


      new UFunc.UImpl[K, CuVector[T], T] {
        def apply(v: CuVector[T]): T = {
          if(v.length > 1024) {
            // TODO TUNE
            val tmpCols = 512
            val tmp = CuVector.create[T](tmpCols)
            kern(tmpCols, (32, 1), 32 * 1 * byteSize.toInt)(v.length, tmp.offsetPointer, v.offsetPointer, v.stride)
            kern(1, (32, 1))(tmp.length, tmp.offsetPointer, tmp.offsetPointer, 1)
            val res = tmp(0 to 0).toDense.apply(0)
            tmp.data.release()
            res
          } else {
            val tmp = CuVector.create[T](1)
            kern(1, (32, 1))(v.length, tmp.offsetPointer, v.offsetPointer, v.stride)
            val res = tmp(0 to 0).toDense.apply(0)
            tmp.data.release()
            res
          }
        }
      }
    }



    def inPlaceImplFor[K <: UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext): UFunc.InPlaceImpl[K, CuVector[T]] = {
      var kern = implCache.get(funName)
      if (kern == null) {
        kern = module.getKernel5[Int, Pointer, Int, Pointer, Int](s"map_${funName}_$typeName")
        implCache.put(funName, kern)
      }


      new UFunc.InPlaceImpl[K, CuVector[T]] {
        def apply(v: CuVector[T]) = {
          kern((512, 1), (32, 1))(v.length, v.offsetPointer, v.stride, v.offsetPointer, v.stride)
        }
      }
    }

    def impl2For[K <: UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext): UFunc.UImpl2[K, CuVector[T], CuVector[T], CuVector[T]] = {
      var kern = impl2Cache.get(funName)
      if (kern == null) {
        kern = module.getKernel7[Int, Pointer, Int, Pointer, Int, Pointer, Int](s"map2_${funName}_$typeName")
        impl2Cache.put(funName, kern)
      }


      new UFunc.UImpl2[K, CuVector[T], CuVector[T], CuVector[T]] {
        def apply(v: CuVector[T], v2: CuVector[T]): CuVector[T] = {
          require(v.length == v2.length, "Dimension mismatch!")
          val res = CuVector.create[T](v.length)
          kern((512, 1), (32, 1))(v.length, res.offsetPointer, res.stride, v.offsetPointer, v.stride, v2.offsetPointer, v2.stride)
          res
        }
      }
    }

    def inPlaceImpl2For[K <: UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext): UFunc.InPlaceImpl2[K, CuVector[T], CuVector[T]] = {
      var kern = impl2Cache.get(funName)
      if (kern == null) {
        kern = module.getKernel7[Int, Pointer, Int, Pointer, Int, Pointer, Int](s"map2_${funName}_$typeName")
        impl2Cache.put(funName, kern)
      }


      new UFunc.InPlaceImpl2[K, CuVector[T], CuVector[T]] {
        def apply(v: CuVector[T], v2: CuVector[T]) {
          require(v.length == v2.length, "Dimension mismatch!")
          kern((512, 1), (32, 1, 1))(v.length, v.offsetPointer, v.stride, v.offsetPointer, v.stride, v2.offsetPointer, v2.stride)
        }
      }
    }

    def impl2For_v_s[K <: UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext): UFunc.UImpl2[K, CuVector[T], T, CuVector[T]] = {
      var kern = impl2VSCache.get(funName)
      if (kern == null) {
        kern = module.getKernel6[Int, Pointer, Int, Pointer, Int, T](s"map2_v_s_${funName}_$typeName")
        impl2VSCache.put(funName, kern)
      }


      new UFunc.UImpl2[K, CuVector[T], T, CuVector[T]] {
        def apply(v: CuVector[T], v2: T): CuVector[T] = {
          val res = CuVector.create[T](v.length)
          kern((512, 20), (32, 1, 1))(v.length, res.offsetPointer, res.stride, v.offsetPointer, v.stride, v2)
          res
        }
      }
    }

    def inPlaceImpl2For_v_s[K <: UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext): UFunc.InPlaceImpl2[K, CuVector[T], T] = {
      var kern = impl2VSCache.get(funName)
      if (kern == null) {
        kern = module.getKernel6[Int, Pointer, Int, Pointer, Int, T](s"map2_v_s_${funName}_$typeName")
        impl2VSCache.put(funName, kern)
      }


      new UFunc.InPlaceImpl2[K, CuVector[T], T] {
        def apply(v: CuVector[T], v2: T) = {
          val res = v
          kern((512, 1), (32, 1, 1))(v.length, res.offsetPointer, res.stride, v.offsetPointer, v.stride, v2)
        }
      }
    }

    def impl2For_s_v[K <: UFunc](funName: String)(implicit context: CuContext = CuContext.ensureContext): UFunc.UImpl2[K, T, CuVector[T], CuVector[T]] = {
      var kern = impl2SVCache.get(funName)
      if (kern == null) {
        kern = module.getKernel6[Int, Pointer, Int, T, Pointer, Int](s"map2_s_v_${funName}_$typeName")
        impl2SVCache.put(funName, kern)
      }


      new UFunc.UImpl2[K, T, CuVector[T], CuVector[T]] {
        def apply(v2: T, v: CuVector[T]): CuVector[T] = {
          val res = CuVector.create[T](v.length)
          kern((512, 1), (32, 1, 1))(v.length, res.offsetPointer, res.stride, v2, v.offsetPointer, v.stride)
          res
        }
      }
    }
  }

}
