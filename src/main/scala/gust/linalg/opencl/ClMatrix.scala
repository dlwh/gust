package gust.linalg.opencl

import java.lang.Math._
import java.nio.ByteOrder

import breeze.linalg.operators.{OpSub, OpAdd, OpSet}
import breeze.linalg.support.CanTranspose
import breeze.linalg.{View, DenseMatrix, NumericOps}
import com.nativelibs4java.opencl.CLMem.Usage
import com.nativelibs4java.opencl._
import com.nativelibs4java.util._
import gust.util.opencl
import gust.util.opencl.CLConfig._
import gust.util.opencl.{CLKernelManager, JInt, JFloat, JLong}
import org.bridj.Pointer
import spire.syntax.cfor._
import scala.collection.JavaConversions._

/**
 * Created by Piotr on 2014-07-13.
 */
class CLMatrix(val rows: Int, val cols: Int,
               val majorStride: Int,
               val data: Pointer[JFloat],
               val buff: CLBuffer[JFloat])(implicit val context: CLContext, val queue: CLQueue) extends NumericOps[CLMatrix] {

  override def repr: CLMatrix = this

  def size = rows * cols

  def linearIndex(row: Int, col: Int) = row + col * majorStride

  def elemSize = data.getIO.getTargetSize

  def indexPointer(row: Int, col: Int) = data.next(linearIndex(row, col))

  def release() = opencl.deallocate(data, buff, context, queue)

  def toDense = {
    val h_A = new DenseMatrix[Float](rows, cols)
    val floats = data.getFloats
    Array.copy(floats, 0, h_A.data, 0, floats.length)
    h_A
  }

  def transpose: CLMatrix = {
    val trans = CLMatrix.create(cols, rows)
    val transposeKernel = CLKernelManager.getKernel("transpose")
    transposeKernel.setArgs(trans.buff, buff, rows.asInstanceOf[JInt],
      cols.asInstanceOf[JInt], trans.majorStride.asInstanceOf[JInt], majorStride.asInstanceOf[JInt])

    val transposeEvt = transposeKernel.enqueueNDRange(queue, Array(rows*cols))
    val outPtr = trans.buff.read(queue, transposeEvt)

    outPtr.copyTo(trans.data)
    trans
  }

  def copy = {
    val c = CLMatrix.create(rows, cols)
    data.copyTo(c.data)
    c
  }

  def writeFromDense(h_A: DenseMatrix[Float]) {
    require(h_A.rows == this.rows, "Matrices must have same number of rows")
    require(h_A.cols == this.cols, "Matrices must have same number of columns")

    val denseDataPtr = Pointer.pointerToArray(h_A.data).as(data.getIO)
    denseDataPtr.copyTo(this.data)
    // there's something odd about this mapping, need to update the buffer as well
    // this is not how this is supposed to work I guess. TODO
    buff.write(queue, data, true)
  }

  def writeFrom(d_A: CLMatrix) {
    require(d_A.rows == this.rows, "Matrices must have same number of rows")
    require(d_A.cols == this.cols, "Matrices must have same number of columns")

    val d_ABuffptr = d_A.buff.map(queue, CLMem.MapFlags.Read)

    d_A.data.copyTo(this.data)
    buff.write(queue, d_ABuffptr, true)

    d_A.buff.unmap(queue, d_ABuffptr)
  }

  /**
   * canReshapeView and reshape are copied almost word-by-word from CuMatrix
   */
  private def canReshapeView = majorStride == rows

  def reshape(rows: Int, cols: Int, view: View=View.Prefer): CLMatrix = {
    require(rows * cols == size, "Cannot reshape a (%d,%d) matrix to a (%d,%d) matrix!".format(this.rows, this.cols, rows, cols))

    view match {
      case View.Require =>
        if(!canReshapeView)
          throw new UnsupportedOperationException("Cannot make a view of this matrix.")
        else
          new CLMatrix(rows, cols, rows, this.data, this.buff)
      case View.Copy =>
        val result = this.copy
        result.reshape(rows, cols, View.Require)
      case View.Prefer =>
        reshape(rows, cols, canReshapeView)
    }
  }

  override def toString = {
    this.toDense.toString()
  }

}

object CLMatrix extends LowPriorityNativeMatrix with CLMatrixOps {

  def create(rows: Int, cols: Int)(implicit context: CLContext, queue: CLQueue) = {
    val (data, buff) = opencl.allocate(rows*cols, context, queue)
    new CLMatrix(rows, cols, rows, data, buff)

  }

  def zeros(rows: Int, cols: Int)(implicit context: CLContext, queue: CLQueue) = {
    val (data, buff) = opencl.allocateZeros(rows*cols, context, queue)
    new CLMatrix(rows, cols, rows, data, buff)
  }

  def fromDense(h_A: DenseMatrix[Float])(implicit context: CLContext, queue: CLQueue) = {
    val (data, buff) = opencl.allocateWithData(h_A.data, context, queue)
    new CLMatrix(h_A.rows, h_A.cols, h_A.majorStride, data, buff)
  }


  implicit def canTranspose: CanTranspose[CLMatrix, CLMatrix] = {
    new CanTranspose[CLMatrix, CLMatrix] {
      def apply(from: CLMatrix) = from.transpose
    }
  }
}

trait LowPriorityNativeMatrix {

  implicit def SetCLMCLMVOp: OpSet.InPlaceImpl2[CLMatrix, CLMatrix] = new OpSet.InPlaceImpl2[CLMatrix, CLMatrix] {
    def apply(a: CLMatrix, b: CLMatrix) {
      a.writeFrom(b)
    }
  }

  implicit def SetCLMDMOp: OpSet.InPlaceImpl2[CLMatrix, DenseMatrix[Float]] = new OpSet.InPlaceImpl2[CLMatrix, DenseMatrix[Float]] {
    def apply(a: CLMatrix, b: DenseMatrix[Float]) {
      a.writeFromDense(b)
    }
  }

}

trait CLMatrixOps {

  implicit def CLMatrixFAddCLMatrixF(implicit context: CLContext, queue: CLQueue): OpAdd.Impl2[CLMatrix, CLMatrix, CLMatrix] = new OpAdd.Impl2[CLMatrix, CLMatrix, CLMatrix] {
    def apply(a : CLMatrix, b : CLMatrix): CLMatrix = {
      require(a.rows == b.rows, s"Row dimension mismatch for addition: ${(a.rows, a.cols)} ${(b.rows, b.cols)}")
      require(a.cols == b.cols, s"Column dimension mismatch: ${(a.rows, a.cols)} ${(b.rows, b.cols)}")

      val blockDim = 8
      val globalDimX = a.rows - (a.rows % 8) + (if (a.rows % 8 != 0) 8 else 0)
      val globalDimY = a.cols - (a.cols % 8) + (if (a.cols % 8 != 0) 8 else 0)

      val addKernel = CLKernelManager.getKernel("add")
      val result = CLMatrix.create(a.rows, a.cols)

      addKernel.setArgs(result.buff, result.majorStride.asInstanceOf[JInt],
                        a.buff, a.majorStride.asInstanceOf[JInt],
                        b.buff, b.majorStride.asInstanceOf[JInt],
                        a.rows.asInstanceOf[JInt], a.cols.asInstanceOf[JInt],
                        blockDim.asInstanceOf[JInt], blockDim.asInstanceOf[JInt])

      val addEvt = addKernel.enqueueNDRange(queue, Array(globalDimX, globalDimY), Array(blockDim, blockDim))
      val outPtr = result.buff.read(queue, addEvt)

      outPtr.copyTo(result.data)
      result
    }
  }

  implicit def CLMatrixFSubCLMatrixF(implicit context: CLContext, queue: CLQueue): OpSub.Impl2[CLMatrix, CLMatrix, CLMatrix] = new OpSub.Impl2[CLMatrix, CLMatrix, CLMatrix] {
    def apply(a: CLMatrix, b: CLMatrix): CLMatrix = {
      require(a.rows == b.rows, s"Row dimension mismatch for subtraction: ${(a.rows, a.cols)} ${(b.rows, b.cols)}")
      require(a.cols == b.cols, s"Column dimension mismatch: ${(a.rows, a.cols)} ${(b.rows, b.cols)}")

      val blockDim = 8
      val globalDimX = a.rows - (a.rows % 8) + (if (a.rows % 8 != 0) 8 else 0)
      val globalDimY = a.cols - (a.cols % 8) + (if (a.cols % 8 != 0) 8 else 0)

      val subKernel = CLKernelManager.getKernel("sub")
      val result = CLMatrix.create(a.rows, a.cols)

      subKernel.setArgs(result.buff, result.majorStride.asInstanceOf[JInt],
        a.buff, a.majorStride.asInstanceOf[JInt],
        b.buff, b.majorStride.asInstanceOf[JInt],
        a.rows.asInstanceOf[JInt], a.cols.asInstanceOf[JInt],
        blockDim.asInstanceOf[JInt], blockDim.asInstanceOf[JInt])

      val subEvt = subKernel.enqueueNDRange(queue, Array(globalDimX, globalDimY), Array(blockDim, blockDim))
      val outPtr = result.buff.read(queue, subEvt)

      outPtr.copyTo(result.data)
      result
    }
  }

  implicit def CLMatrixFAddCLMatrixFInPlace(implicit context: CLContext, queue: CLQueue): OpAdd.InPlaceImpl2[CLMatrix, CLMatrix] = new OpAdd.InPlaceImpl2[CLMatrix, CLMatrix] {
    def apply(a : CLMatrix, b : CLMatrix): Unit = {
      require(a.rows == b.rows, s"Row dimension mismatch for addition: ${(a.rows, a.cols)} ${(b.rows, b.cols)}")
      require(a.cols == b.cols, s"Column dimension mismatch: ${(a.rows, a.cols)} ${(b.rows, b.cols)}")

      val blockDim = 8
      val globalDimX = a.rows - (a.rows % 8) + (if (a.rows % 8 != 0) 8 else 0)
      val globalDimY = a.cols - (a.cols % 8) + (if (a.cols % 8 != 0) 8 else 0)

      val addKernel = CLKernelManager.getKernel("add_in_place")
      addKernel.setArgs(a.buff, a.majorStride.asInstanceOf[JInt],
        b.buff, b.majorStride.asInstanceOf[JInt],
        a.rows.asInstanceOf[JInt], a.cols.asInstanceOf[JInt],
        blockDim.asInstanceOf[JInt], blockDim.asInstanceOf[JInt])

      val addEvt = addKernel.enqueueNDRange(queue, Array(globalDimX, globalDimY), Array(blockDim, blockDim))
      val outPtr = a.buff.read(queue, addEvt)

      outPtr.copyTo(a.data)
    }
  }

  implicit def CLMatrixFSubCLMatrixFInPlace(implicit context: CLContext, queue: CLQueue): OpSub.InPlaceImpl2[CLMatrix, CLMatrix] = new OpSub.InPlaceImpl2[CLMatrix, CLMatrix] {
    def apply(a : CLMatrix, b : CLMatrix): Unit = {
      require(a.rows == b.rows, s"Row dimension mismatch for addition: ${(a.rows, a.cols)} ${(b.rows, b.cols)}")
      require(a.cols == b.cols, s"Column dimension mismatch: ${(a.rows, a.cols)} ${(b.rows, b.cols)}")

      val blockDim = 8
      val globalDimX = a.rows - (a.rows % 8) + (if (a.rows % 8 != 0) 8 else 0)
      val globalDimY = a.cols - (a.cols % 8) + (if (a.cols % 8 != 0) 8 else 0)

      val subKernel = CLKernelManager.getKernel("sub_in_place")
      subKernel.setArgs(a.buff, a.majorStride.asInstanceOf[JInt],
                        b.buff, b.majorStride.asInstanceOf[JInt],
                        a.rows.asInstanceOf[JInt], a.cols.asInstanceOf[JInt],
                        blockDim.asInstanceOf[JInt], blockDim.asInstanceOf[JInt])

      val subEvt = subKernel.enqueueNDRange(queue, Array(globalDimX, globalDimY), Array(blockDim, blockDim))
      val outPtr = a.buff.read(queue, subEvt)

      outPtr.copyTo(a.data)
    }
  }

}
