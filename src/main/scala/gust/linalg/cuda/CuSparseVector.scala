package gust.linalg.cuda

import breeze.linalg.NumericOps
import gust.util.cuda.{CuContext, CuDevice}
import jcuda.driver.{CUfunction, CUmodule, JCudaDriver}
import jcuda.jcublas.cublasHandle
import jcuda.jcusparse.{cusparseHandle, cusparseMatDescr}
import org.bridj.Pointer
import gust.util.cuda._

import scala.reflect.ClassTag

/**
 * Copyright 2014 Piotr Moczurad
 *
 */
class CuSparseVector[V](val length: Int,
                        val data: CuMatrix[V],
                        val indices: CuMatrix[Int]) extends NumericOps[CuSparseVector[V]] {
  override def repr: CuSparseVector[V] = this

  def size = length
  def nnz = data.rows
  def elemSize = data.elemSize

  def toCuVector(implicit ct: ClassTag[V]): CuVector[V] = {
    val dv = CuVector.zeros[V](length)

    val nb = 512

    JCudaDriver.setExceptionsEnabled(true)
    implicit val dev = CuDevice(0)
    val ctx = CuContext.ensureContext
    val module = new CUmodule()
    val zero_out = new CUfunction()
    JCudaDriver.cuModuleLoad(module, "src/main/resources/gust/linalg/cuda/sparseVecKernels.ptx")

    val funcName = if (elemSize == 8) "sparse2dense_double" else if (elemSize == 4) "sparse2dense_float"
                   else throw new UnsupportedOperationException("Can only convert with elem sizes 4 or 8")
    JCudaDriver.cuModuleGetFunction(zero_out, module, funcName)

    // kernel parameters:
    val nnzPtr = Pointer.pointerToInt(nnz).toCuPointer

    val params = jcuda.Pointer.to(
      jcuda.Pointer.to(dv.offsetPointer),
      jcuda.Pointer.to(data.offsetPointer),
      jcuda.Pointer.to(indices.offsetPointer),
      nnzPtr
    )

    val blockNum = nnz / nb + (if (nnz % nb == 0) 0 else 1)
    val threadNum = nb

    JCudaDriver.cuLaunchKernel(zero_out, blockNum, 1, 1,
      threadNum, 1, 1,
      0, null, params, null)
    JCudaDriver.cuCtxSynchronize()

    dv
  }

  override def toString = "CuSparseVector {\nData: " + data.t.toString + "indices" + indices.t.toString + "}"
}

object CuSparseVector {
  def fromCuVector[V <: AnyVal](cv: CuVector[V])(implicit ct: ClassTag[V], handle: cublasHandle, sph: cusparseHandle) = {
    val cspm = CuSparseMatrix.fromCuMatrix(CuMatrix.fromCuVector(cv))

    new CuSparseVector[V](cv.length, cspm.cscVal, cspm.cscRowInd)
  }
}
