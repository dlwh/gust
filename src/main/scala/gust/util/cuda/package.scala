package gust.util

import scala.reflect.ClassTag
import org.bridj.{PointerIO, Pointer}
import jcuda.runtime.{cudaStream_t, JCuda}
import jcuda.jcublas.{cublasOperation, JCublas2, cublasHandle}
import jcuda.driver.{JCudaDriver, CUcontext, CUfunction, CUstream}
import jcuda.NativePointerObject
import breeze.macros.arityize

/**
 * TODO
 *
 * @author dlwh
 **/
package object cuda {
  type CuPointer = jcuda.Pointer

  def allocate[V:ClassTag](size: Long) = {
    val ptr = new CuPointer()
    val tpe = implicitly[ClassTag[V]].runtimeClass
    val io = PointerIO.getInstance[V](tpe)
    JCuda.cudaMalloc(ptr, size * io.getTargetSize)
    Pointer.pointerToAddress(nativePtr(ptr), size, io, DeviceFreeReleaser)
  }

  def allocateHost[V:ClassTag](size: Long) = {
    val ptr = new CuPointer()
    val tpe = implicitly[ClassTag[V]].runtimeClass
    val io = PointerIO.getInstance[V](tpe)
    JCuda.cudaMallocHost(ptr, size * io.getTargetSize)
    Pointer.pointerToAddress(nativePtr(ptr), size, io, HostFreeReleaser)
  }

  def cuPointerToArray[T](array: Array[T]): jcuda.Pointer = array match {
    case array: Array[Int] => jcuda.Pointer.to(array)
    case array: Array[Byte] => jcuda.Pointer.to(array)
    case array: Array[Long] => jcuda.Pointer.to(array)
    case array: Array[Short] => jcuda.Pointer.to(array)
    case array: Array[Float] => jcuda.Pointer.to(array)
    case array: Array[Double] => jcuda.Pointer.to(array)
    case _ => throw new UnsupportedOperationException("Can't deal with this array type!")
  }

  implicit class enrichBridjPtr[T](val pointer: Pointer[T]) extends AnyVal {
    def toCuPointer = {
      fromNativePtr(pointer.getPeer)
    }
  }

  private object DeviceFreeReleaser extends Pointer.Releaser {
    def release(p: Pointer[_]): Unit = {
      val ptr = fromNativePtr(p.getPeer)
      JCuda.cudaFree(ptr)
    }
  }

  private object HostFreeReleaser extends Pointer.Releaser {
    def release(p: Pointer[_]): Unit = {
      val ptr = fromNativePtr(p.getPeer)
      JCuda.cudaFreeHost(ptr)
    }
  }


  private def nativePtr(pointer: CuPointer) = {
    val m = classOf[NativePointerObject].getDeclaredMethod("getNativePointer")
    m.setAccessible(true)
    m.invoke(pointer).asInstanceOf[java.lang.Long].longValue()
  }

  private def fromNativePtr(peer: Long, offset: Long = 0) = {
    val m = classOf[CuPointer].getDeclaredConstructor(java.lang.Long.TYPE)
    m.setAccessible(true)
    m.newInstance(java.lang.Long.valueOf(peer)).withByteOffset(offset)
  }

  implicit def cudaStreamToCuStream(s: CUstream) = new cudaStream_t(s)

  implicit class richBlas(val blas: cublasHandle) extends AnyVal {
    def withStream[T](stream: cudaStream_t)(block: => T) = blas.synchronized {
      val olds = new cudaStream_t()
      JCublas2.cublasGetStream(blas, olds)
      JCublas2.cublasSetStream(blas, stream)
      val res = block
      JCublas2.cublasSetStream(blas, olds)
      res
    }
  }


  private[gust] val contextLock = new Object()

  @arityize(10)
  class CuKernel[@arityize.replicate T](module: CuModule, fn: CUfunction) {
    def apply(gridDims: Dim3 = Dim3.default, blockDims: Dim3 = Dim3.default, sharedMemorySize: Int = 0)(@arityize.replicate t: T @arityize.relative(t))(implicit context: CuContext):Unit = {
      CuKernel.invoke(fn, gridDims, blockDims, sharedMemorySize)((t: @arityize.replicate ))
    }
  }
}
