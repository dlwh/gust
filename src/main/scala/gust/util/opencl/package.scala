package gust.util

import com.nativelibs4java.opencl.CLMem.Usage
import com.nativelibs4java.opencl.{CLMem, CLQueue, CLBuffer, CLContext}
import org.bridj.{Pointer, PointerIO}

import scala.reflect.ClassTag

package object opencl {

  type JFloat = java.lang.Float
  type JLong = java.lang.Long
  type JInt = java.lang.Integer

  def allocate(size: Long, context: CLContext, queue: CLQueue) = {
    val byteOrder = context.getByteOrder
    //val tpe = implicitly[ClassTag[V]].runtimeClass
    //val io = PointerIO.getInstance[V](tpe)
    val ptr = Pointer.allocateFloats(size).order(byteOrder)
    val buff = context.createBuffer(Usage.InputOutput, ptr, false) // host allocation

    //val data: Pointer[JFloat] = buff.map(queue, CLMem.MapFlags.ReadWrite)
    (ptr, buff)
  }

  def allocateZeros(size: Long, context: CLContext, queue: CLQueue)  = {
    val byteOrder = context.getByteOrder
    val ptr = Pointer.allocateFloats(size).order(byteOrder)
    ptr.clearValidBytes() // I don't really know if it's the right thing to use
    val buff = context.createBuffer(Usage.InputOutput, ptr, false) // host allocation

    //val data: Pointer[JFloat] = buff.map(queue, CLMem.MapFlags.ReadWrite)
    (ptr, buff)
  }

  def allocateWithData(array: Array[Float], context: CLContext, queue: CLQueue) = {
    val byteOrder = context.getByteOrder
    val ptr: Pointer[JFloat] = Pointer.pointerToArray(array).as(java.lang.Float.TYPE)

    val buff = context.createBuffer(Usage.InputOutput, ptr, false) // host allocation

    //val data: Pointer[JFloat] = buff.map(queue, CLMem.MapFlags.ReadWrite)
    (ptr, buff)
  }

  def deallocate(pointer: Pointer[JFloat], buffer: CLBuffer[JFloat], context: CLContext, queue: CLQueue) = {
    //buffer.unmap(queue, pointer)
    pointer.release()
  }

  implicit def int2Jint(n: Int) = n.asInstanceOf[JInt]

}
