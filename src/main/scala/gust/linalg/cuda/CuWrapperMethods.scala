package gust.linalg.cuda

import gust.util.cuda
import cuda._
import jcuda.jcublas.{cublasOperation, JCublas2, cublasHandle}
import breeze.linalg.DenseMatrix
import jcuda.jcusparse._
import jcuda.runtime.{cudaMemcpyKind, JCuda}
import jcuda.driver.{CUfunction, CUmodule, JCudaDriver}
import gust.util.cuda.{CuContext, CuDevice}
import org.bridj.Pointer
import spire.syntax.cfor._
import scala.reflect.ClassTag


/**
 * Copyright 2014 Piotr Moczurad
 *
 */
object CuWrapperMethods {

  /*
   * wrapped calls to kernels for zeroing out some parts of matrices.
   */
  def zeroOutFloat(A: CuMatrix[Float], fillMode: Char, includeDiagonal: Boolean = false) {
    val nb = 32

    JCudaDriver.setExceptionsEnabled(true)
    implicit val dev = CuDevice(0)
    val ctx = CuContext.ensureContext
    val module = new CUmodule()
    val zero_out = new CUfunction()
    JCudaDriver.cuModuleLoad(module, "src/main/resources/gust/linalg/cuda/enforceLUFloat.ptx")

    val funcName = if (fillMode == 'U') "zerosU" else "zerosL"
    JCudaDriver.cuModuleGetFunction(zero_out, module, funcName)

    // kernel parameters:
    val ldaArr = Array(A.majorStride)
    val lda = jcuda.Pointer.to(ldaArr)
    //val elemsArr = Array(A.rows * A.cols)
    //val elems = jcuda.Pointer.to(elemsArr)
    val mArr = Array(A.rows)
    val m = jcuda.Pointer.to(mArr)
    val nArr = Array(A.cols)
    val n = jcuda.Pointer.to(nArr)
    val inclArr = Array(if (includeDiagonal) 1 else 0)
    val incl = jcuda.Pointer.to(inclArr)

    val params = jcuda.Pointer.to(
      m, n,
      jcuda.Pointer.to(A.offsetPointer),
      lda, incl
    )

    val gridDim = (A.rows / nb + (if (A.rows % nb == 0) 0 else 1),
                   A.cols / nb + (if (A.cols % nb == 0) 0 else 1),
                   1)
    val blockDim = (nb, nb, 1)

    JCudaDriver.cuLaunchKernel(zero_out, gridDim._1, gridDim._2, gridDim._3,
                                         blockDim._1, blockDim._2, blockDim._3,
                                         0, null, params, null)
    JCudaDriver.cuCtxSynchronize()
  }

  def zeroOutDouble(A: CuMatrix[Double], fillMode: Char, includeDiagonal: Boolean = false) {
    val nb = 32

    JCudaDriver.setExceptionsEnabled(true)
    implicit val dev = CuDevice(0)
    val ctx = CuContext.ensureContext
    val module = new CUmodule()
    JCudaDriver.cuModuleLoad(module, "src/main/resources/gust/linalg/cuda/enforceLUDouble.ptx")

    val zero_out = new CUfunction()
    val funcName = if (fillMode == 'U') "zerosU" else "zerosL"
    JCudaDriver.cuModuleGetFunction(zero_out, module, funcName)

    // kernel parameters:
    val ldaArr = Array(A.majorStride)
    val lda = jcuda.Pointer.to(ldaArr)
    //val elemsArr = Array(A.rows * A.cols)
    //val elems = jcuda.Pointer.to(elemsArr)
    val mArr = Array(A.rows)
    val m = jcuda.Pointer.to(mArr)
    val nArr = Array(A.cols)
    val n = jcuda.Pointer.to(nArr)
    val inclArr = Array(if (includeDiagonal) 1 else 0)
    val incl = jcuda.Pointer.to(inclArr)

    val params = jcuda.Pointer.to(
      m, n,
      jcuda.Pointer.to(A.offsetPointer),
      lda, incl
    )

    val gridDim = (A.rows / nb + (if (A.rows % nb == 0) 0 else 1),
      A.cols / nb + (if (A.cols % nb == 0) 0 else 1), 1)
    val blockDim = (nb, nb, 1)

    JCudaDriver.cuLaunchKernel(zero_out,
      gridDim._1, gridDim._2, gridDim._3,
      blockDim._1, blockDim._2, blockDim._3,
      0, null, params, null)

    JCudaDriver.cuCtxSynchronize()
  }

  def zeroOutFloatOffset(A: CuMatrix[Float], Aroff: Int, Acoff: Int, fillMode: Char, includeDiagonal: Boolean = false) {
    val nb = 32

    JCudaDriver.setExceptionsEnabled(true)
    implicit val dev = CuDevice(0)
    val ctx = CuContext.ensureContext
    val module = new CUmodule()
    val zero_out = new CUfunction()
    JCudaDriver.cuModuleLoad(module, "src/main/resources/gust/linalg/cuda/enforceLUFloat.ptx")

    val funcName = if (fillMode == 'U') "zerosU" else "zerosL"
    JCudaDriver.cuModuleGetFunction(zero_out, module, funcName)

    // kernel parameters:
    val ldaArr = Array(A.majorStride)
    val lda = jcuda.Pointer.to(ldaArr)
    //val elemsArr = Array(A.rows * A.cols)
    //val elems = jcuda.Pointer.to(elemsArr)
    val mArr = Array(A.rows - Aroff)
    val m = jcuda.Pointer.to(mArr)
    val nArr = Array(A.cols - Acoff)
    val n = jcuda.Pointer.to(nArr)
    val inclArr = Array(if (includeDiagonal) 1 else 0)
    val incl = jcuda.Pointer.to(inclArr)

    val params = jcuda.Pointer.to(
      m, n,
      jcuda.Pointer.to(A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff) * A.elemSize)),
      lda, incl
    )

    val gridDim = ((A.rows - Aroff) / nb + (if ((A.rows - Aroff) % nb == 0) 0 else 1),
      (A.cols - Acoff) / nb + (if ((A.cols - Acoff) % nb == 0) 0 else 1),
      1)
    val blockDim = (nb, nb, 1)

    JCudaDriver.cuLaunchKernel(zero_out, gridDim._1, gridDim._2, gridDim._3,
      blockDim._1, blockDim._2, blockDim._3,
      0, null, params, null)
    JCudaDriver.cuCtxSynchronize()
  }

  def zeroOutDoubleOffset(A: CuMatrix[Double], Aroff: Int, Acoff: Int, fillMode: Char, includeDiagonal: Boolean = false) {
    val nb = 32     // most cards can go as high as 1024 (32**2) threads per block

    JCudaDriver.setExceptionsEnabled(true)
    implicit val dev = CuDevice(0)
    val ctx = CuContext.ensureContext
    val module = new CUmodule()
    val zero_out = new CUfunction()
    JCudaDriver.cuModuleLoad(module, "src/main/resources/gust/linalg/cuda/enforceLUDouble.ptx")

    val funcName = if (fillMode == 'U') "zerosU" else "zerosL"
    JCudaDriver.cuModuleGetFunction(zero_out, module, funcName)

    // kernel parameters:
    val ldaArr = Array(A.majorStride)
    val lda = jcuda.Pointer.to(ldaArr)
    val mArr = Array(A.rows - Aroff)
    val m = jcuda.Pointer.to(mArr)
    val nArr = Array(A.cols - Acoff)
    val n = jcuda.Pointer.to(nArr)
    val inclArr = Array(if (includeDiagonal) 1 else 0)
    val incl = jcuda.Pointer.to(inclArr)

    val params = jcuda.Pointer.to(
      m, n,
      jcuda.Pointer.to(A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff) * A.elemSize)),
      lda, incl
    )

    val gridDim = ((A.rows - Aroff) / nb + (if ((A.rows - Aroff) % nb == 0) 0 else 1),
      (A.cols - Acoff) / nb + (if ((A.cols - Acoff) % nb == 0) 0 else 1),
      1)
    val blockDim = (nb, nb, 1)

    JCudaDriver.cuLaunchKernel(zero_out, gridDim._1, gridDim._2, gridDim._3,
      blockDim._1, blockDim._2, blockDim._3,
      0, null, params, null)
    JCudaDriver.cuCtxSynchronize()
  }

  /*
   * wrapped calls to gemm, don't require passing lda's and elemSizes around
   */
  def SgemmNN(m: Int, n: Int, k: Int, alpha: jcuda.Pointer,
              A: CuMatrix[Float], Aroff: Int, Acoff: Int,
              B: CuMatrix[Float], Broff: Int, Bcoff: Int,
              beta: jcuda.Pointer,
              C: CuMatrix[Float], Croff: Int, Ccoff: Int)(implicit handle: cublasHandle): Int = {

    JCublas2.cublasSgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
      m, n, k, alpha,
      A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff) * A.elemSize), A.majorStride,
      B.offsetPointer.withByteOffset(B.linearIndex(Broff, Bcoff) * B.elemSize), B.majorStride,
      beta,
      C.offsetPointer.withByteOffset(C.linearIndex(Croff, Ccoff) * C.elemSize), C.majorStride)
  }

  def DgemmNN(m: Int, n: Int, k: Int, alpha: jcuda.Pointer,
              A: CuMatrix[Double], Aroff: Int, Acoff: Int,
              B: CuMatrix[Double], Broff: Int, Bcoff: Int,
              beta: jcuda.Pointer,
              C: CuMatrix[Double], Croff: Int, Ccoff: Int)(implicit handle: cublasHandle): Int = {

    JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
      m, n, k, alpha,
      A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff) * A.elemSize), A.majorStride,
      B.offsetPointer.withByteOffset(B.linearIndex(Broff, Bcoff) * B.elemSize), B.majorStride,
      beta,
      C.offsetPointer.withByteOffset(C.linearIndex(Croff, Ccoff) * C.elemSize), C.majorStride)
  }

  def SgemmNT(m: Int, n: Int, k: Int, alpha: jcuda.Pointer,
              A: CuMatrix[Float], Aroff: Int, Acoff: Int,
              B: CuMatrix[Float], Broff: Int, Bcoff: Int,
              beta: jcuda.Pointer,
              C: CuMatrix[Float], Croff: Int, Ccoff: Int)(implicit handle: cublasHandle): Int = {

    JCublas2.cublasSgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T,
      m, n, k, alpha,
      A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff) * A.elemSize), A.majorStride,
      B.offsetPointer.withByteOffset(B.linearIndex(Broff, Bcoff) * B.elemSize), B.majorStride,
      beta,
      C.offsetPointer.withByteOffset(C.linearIndex(Croff, Ccoff) * C.elemSize), C.majorStride)
  }

  def DgemmNT(m: Int, n: Int, k: Int, alpha: jcuda.Pointer,
              A: CuMatrix[Double], Aroff: Int, Acoff: Int,
              B: CuMatrix[Double], Broff: Int, Bcoff: Int,
              beta: jcuda.Pointer,
              C: CuMatrix[Double], Croff: Int, Ccoff: Int)(implicit handle: cublasHandle): Int = {

    JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T,
      m, n, k, alpha,
      A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff) * A.elemSize), A.majorStride,
      B.offsetPointer.withByteOffset(B.linearIndex(Broff, Bcoff) * B.elemSize), B.majorStride,
      beta,
      C.offsetPointer.withByteOffset(C.linearIndex(Croff, Ccoff) * C.elemSize), C.majorStride)
  }

  def SgemmTN(m: Int, n: Int, k: Int, alpha: jcuda.Pointer,
              A: CuMatrix[Float], Aroff: Int, Acoff: Int,
              B: CuMatrix[Float], Broff: Int, Bcoff: Int,
              beta: jcuda.Pointer,
              C: CuMatrix[Float], Croff: Int, Ccoff: Int)(implicit handle: cublasHandle): Int = {

    JCublas2.cublasSgemm(handle, cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N,
      m, n, k, alpha,
      A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff) * A.elemSize), A.majorStride,
      B.offsetPointer.withByteOffset(B.linearIndex(Broff, Bcoff) * B.elemSize), B.majorStride,
      beta,
      C.offsetPointer.withByteOffset(C.linearIndex(Croff, Ccoff) * C.elemSize), C.majorStride)
  }

  def DgemmTN(m: Int, n: Int, k: Int, alpha: jcuda.Pointer,
              A: CuMatrix[Double], Aroff: Int, Acoff: Int,
              B: CuMatrix[Double], Broff: Int, Bcoff: Int,
              beta: jcuda.Pointer,
              C: CuMatrix[Double], Croff: Int, Ccoff: Int)(implicit handle: cublasHandle): Int = {

    JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N,
      m, n, k, alpha,
      A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff) * A.elemSize), A.majorStride,
      B.offsetPointer.withByteOffset(B.linearIndex(Broff, Bcoff) * B.elemSize), B.majorStride,
      beta,
      C.offsetPointer.withByteOffset(C.linearIndex(Croff, Ccoff) * C.elemSize), C.majorStride)
  }

  def SgemmTT(m: Int, n: Int, k: Int, alpha: jcuda.Pointer,
              A: CuMatrix[Float], Aroff: Int, Acoff: Int,
              B: CuMatrix[Float], Broff: Int, Bcoff: Int,
              beta: jcuda.Pointer,
              C: CuMatrix[Float], Croff: Int, Ccoff: Int)(implicit handle: cublasHandle): Int = {

    JCublas2.cublasSgemm(handle, cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_T,
      m, n, k, alpha,
      A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff) * A.elemSize), A.majorStride,
      B.offsetPointer.withByteOffset(B.linearIndex(Broff, Bcoff) * B.elemSize), B.majorStride,
      beta,
      C.offsetPointer.withByteOffset(C.linearIndex(Croff, Ccoff) * C.elemSize), C.majorStride)
  }

  def DgemmTT(m: Int, n: Int, k: Int, alpha: jcuda.Pointer,
              A: CuMatrix[Double], Aroff: Int, Acoff: Int,
              B: CuMatrix[Double], Broff: Int, Bcoff: Int,
              beta: jcuda.Pointer,
              C: CuMatrix[Double], Croff: Int, Ccoff: Int)(implicit handle: cublasHandle): Int = {

    JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_T,
      m, n, k, alpha,
      A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff) * A.elemSize), A.majorStride,
      B.offsetPointer.withByteOffset(B.linearIndex(Broff, Bcoff) * B.elemSize), B.majorStride,
      beta,
      C.offsetPointer.withByteOffset(C.linearIndex(Croff, Ccoff) * C.elemSize), C.majorStride)
  }

  /*
   * Methods for moving matrices between GPU and CPU
   * Could use get/setMatrix but memcpy works fine
   */
  def uploadFloat(m: Int, n: Int, dst: CuMatrix[Float],
                  dst_roff: Int, dst_coff: Int,
                  src: DenseMatrix[Float], src_roff: Int, src_coff: Int): Int = {
    JCuda.cudaMemcpy2D(dst.offsetPointer.withByteOffset(dst.linearIndex(dst_roff, dst_coff) * dst.elemSize),
      dst.majorStride * dst.elemSize,
      jcuda.Pointer.to(src.data).withByteOffset(src.linearIndex(src_roff, src_coff) * dst.elemSize),
      src.majorStride * dst.elemSize, m * dst.elemSize, n,
      cudaMemcpyKind.cudaMemcpyHostToDevice)
  }

  def uploadDouble(m: Int, n: Int, dst: CuMatrix[Double],
                   dst_roff: Int, dst_coff: Int,
                   src: DenseMatrix[Double], src_roff: Int, src_coff: Int): Int = {
    JCuda.cudaMemcpy2D(dst.offsetPointer.withByteOffset(dst.linearIndex(dst_roff, dst_coff) * dst.elemSize),
      dst.majorStride * dst.elemSize,
      jcuda.Pointer.to(src.data).withByteOffset(src.linearIndex(src_roff, src_coff) * dst.elemSize),
      src.majorStride * dst.elemSize, m * dst.elemSize, n,
      cudaMemcpyKind.cudaMemcpyHostToDevice)
  }

  def downloadFloat(m: Int, n: Int, dst: DenseMatrix[Float], dst_roff: Int, dst_coff: Int,
                    src: CuMatrix[Float], src_roff: Int, src_coff: Int): Int = {
    JCuda.cudaMemcpy2D(jcuda.Pointer.to(dst.data).withByteOffset(dst.linearIndex(dst_roff, dst_coff) * src.elemSize),
      dst.majorStride * src.elemSize,
      src.offsetPointer.withByteOffset(src.linearIndex(src_roff, src_coff) * src.elemSize),
      src.majorStride * src.elemSize, m * src.elemSize, n, cudaMemcpyKind.cudaMemcpyDeviceToHost)
  }

  def downloadDouble(m: Int, n: Int, dst: DenseMatrix[Double], dst_roff: Int, dst_coff: Int,
                     src: CuMatrix[Double], src_roff: Int, src_coff: Int): Int = {
    JCuda.cudaMemcpy2D(jcuda.Pointer.to(dst.data).withByteOffset(dst.linearIndex(dst_roff, dst_coff) * src.elemSize),
      dst.majorStride * src.elemSize,
      src.offsetPointer.withByteOffset(src.linearIndex(src_roff, src_coff) * src.elemSize),
      src.majorStride * src.elemSize, m * src.elemSize, n, cudaMemcpyKind.cudaMemcpyDeviceToHost)
  }

  def copyFloat(m: Int, n: Int, dst: CuMatrix[Float], dst_roff: Int, dst_coff: Int,
                src: CuMatrix[Float], src_roff: Int, src_coff: Int) {
    copy[Float](m, n, dst, dst_roff, dst_coff, src, src_roff, src_coff, "src/main/resources/gust/linalg/cuda/elemWiseFloat.ptx")
  }

  def copyDouble(m: Int, n: Int, dst: CuMatrix[Double], dst_roff: Int, dst_coff: Int,
                src: CuMatrix[Double], src_roff: Int, src_coff: Int) {
    copy[Double](m, n, dst, dst_roff, dst_coff, src, src_roff, src_coff, "src/main/resources/gust/linalg/cuda/elemWiseDouble.ptx")
  }

  def copy[V](m: Int, n: Int, dst: CuMatrix[V], dst_roff: Int, dst_coff: Int,
                 src: CuMatrix[V], src_roff: Int, src_coff: Int, kernelPath: String) {
    val nb = 32     // most cards can go as high as 1024 (32**2) threads per block

    JCudaDriver.setExceptionsEnabled(true)
    implicit val dev = CuDevice(0)
    val ctx = CuContext.ensureContext
    val module = new CUmodule()
    val copy = new CUfunction()
    JCudaDriver.cuModuleLoad(module, kernelPath)

    val funcName = "copy"
    JCudaDriver.cuModuleGetFunction(copy, module, funcName)

    // kernel parameters:
    val ldsrcArr = Array(src.majorStride)
    val ldsrc = jcuda.Pointer.to(ldsrcArr)
    val lddstArr = Array(dst.majorStride)
    val lddst = jcuda.Pointer.to(lddstArr)


    val mArr = Array(m)
    val _m = jcuda.Pointer.to(mArr)
    val nArr = Array(n)
    val _n = jcuda.Pointer.to(nArr)

    val params = jcuda.Pointer.to(
      _m, _n,
      jcuda.Pointer.to(dst.offsetPointer.withByteOffset(dst.linearIndex(dst_roff, dst_coff) * dst.elemSize)),
      lddst,
      jcuda.Pointer.to(src.offsetPointer.withByteOffset(src.linearIndex(src_roff, src_coff) * src.elemSize)),
      ldsrc
    )

    val gridDim = ((src.rows - src_roff) / nb + (if ((src.rows - src_roff) % nb == 0) 0 else 1),
      (src.cols - src_coff) / nb + (if ((src.cols - src_coff) % nb == 0) 0 else 1),
      1)
    val blockDim = (nb, nb, 1)

    JCudaDriver.cuLaunchKernel(copy, gridDim._1, gridDim._2, gridDim._3,
      blockDim._1, blockDim._2, blockDim._3,
      0, null, params, null)
    JCudaDriver.cuCtxSynchronize()
  }


  /*
   * Functions for calculating the residuals:
   * calculates |A - BC| where BC are Q and R in case of QR factorization or L and U
   * in case of LU factorization.
   * If the pivot matrix is not null, we actually calculate |PA - BC|.
   * In case of Cholesky factorization L and L.t should be given as B and C
   *
   * It can also calculate the residual in case of the solve method, since we treat vectors as matrices
   */
  def residualFloat(A: CuMatrix[Float], B: CuMatrix[Float], C: CuMatrix[Float], P: CuMatrix[Float] = null)(implicit handle: cublasHandle): Double = {
    if (B.rows != C.cols) {
      println("Dimensions have to match (B.rows must equal C.cols)")
      return 0.0
    }

    if (B.cols != C.rows) {
      println("Dimensions have to match (B.cols must equal C.rows)")
      return 0.0
    }

    if (A.rows != C.rows) {
      println("Dimensions have to match (A.rows must equal C.rows)")
      return 0.0
    }

    if (P != null && (A.rows != P.rows || A.cols != P.cols)) {
      println("Wrong pivoting matrix")
      return 0.0
    }

    val d_A = CuMatrix.create[Float](A.rows, A.cols)

    val minusOneArr = Array(-1.0f)
    val oneArr = Array(1.0f)
    val zeroArr = Array(0.0f)

    if (P != null)
      // d_A = P*A
      if (!A.isTranspose)
        SgemmNN(A.rows, A.cols, A.cols, jcuda.Pointer.to(oneArr), P, 0, 0, A, 0, 0, jcuda.Pointer.to(zeroArr), d_A, 0, 0)
      else
        SgemmNT(A.rows, A.cols, A.cols, jcuda.Pointer.to(oneArr), P, 0, 0, A, 0, 0, jcuda.Pointer.to(zeroArr), d_A, 0, 0)
    else
      d_A := A

    // d_A = d_A - B*C
    if (B.isTranspose && C.isTranspose)
      SgemmTT(d_A.rows, d_A.cols, C.cols, jcuda.Pointer.to(minusOneArr), B, 0, 0, C, 0, 0, jcuda.Pointer.to(oneArr), d_A, 0, 0)
    else if (B.isTranspose)
      SgemmTN(d_A.rows, d_A.cols, C.cols, jcuda.Pointer.to(minusOneArr), B, 0, 0, C, 0, 0, jcuda.Pointer.to(oneArr), d_A, 0, 0)
    else if (C.isTranspose)
      SgemmNT(d_A.rows, d_A.cols, C.cols, jcuda.Pointer.to(minusOneArr), B, 0, 0, C, 0, 0, jcuda.Pointer.to(oneArr), d_A, 0, 0)
    else
      SgemmNN(d_A.rows, d_A.cols, C.cols, jcuda.Pointer.to(minusOneArr), B, 0, 0, C, 0, 0, jcuda.Pointer.to(oneArr), d_A, 0, 0)

    d_A.norm
  }

  def residualDouble(A: CuMatrix[Double], B: CuMatrix[Double], C: CuMatrix[Double], P: CuMatrix[Double] = null)(implicit handle: cublasHandle): Double = {
    if (B.rows != C.cols) {
      println("Dimensions have to match (B.rows must equal C.cols)")
      return 0.0
    }

    if (B.cols != C.rows) {
      println("Dimensions have to match (B.cols must equal C.rows)")
      return 0.0
    }

    if (A.rows != C.rows) {
      println("Dimensions have to match (A.rows must equal C.rows)")
      return 0.0
    }

    if (P != null && (A.rows != P.rows || A.cols != P.cols)) {
      println("Wrong pivoting matrix")
      return 0.0
    }

    val d_A = CuMatrix.create[Double](A.rows, A.cols)

    val minusOneArr = Array(-1.0)
    val oneArr = Array(1.0)
    val zeroArr = Array(0.0)

    if (P != null)
    // d_A = P*A
      if (!A.isTranspose)
        DgemmNN(A.rows, A.cols, A.cols, jcuda.Pointer.to(oneArr), P, 0, 0, A, 0, 0, jcuda.Pointer.to(zeroArr), d_A, 0, 0)
      else
        DgemmNT(A.rows, A.cols, A.cols, jcuda.Pointer.to(oneArr), P, 0, 0, A, 0, 0, jcuda.Pointer.to(zeroArr), d_A, 0, 0)
    else
      d_A := A

    // d_A = d_A - B*C
    if (B.isTranspose && C.isTranspose)
      DgemmTT(d_A.rows, d_A.cols, C.cols, jcuda.Pointer.to(minusOneArr), B, 0, 0, C, 0, 0, jcuda.Pointer.to(oneArr), d_A, 0, 0)
    else if (B.isTranspose)
      DgemmTN(d_A.rows, d_A.cols, C.cols, jcuda.Pointer.to(minusOneArr), B, 0, 0, C, 0, 0, jcuda.Pointer.to(oneArr), d_A, 0, 0)
    else if (C.isTranspose)
      DgemmNT(d_A.rows, d_A.cols, C.cols, jcuda.Pointer.to(minusOneArr), B, 0, 0, C, 0, 0, jcuda.Pointer.to(oneArr), d_A, 0, 0)
    else
      DgemmNN(d_A.rows, d_A.cols, C.cols, jcuda.Pointer.to(minusOneArr), B, 0, 0, C, 0, 0, jcuda.Pointer.to(oneArr), d_A, 0, 0)

    d_A.norm
  }

  /**
   * Overwrites matrix A with an identity matrix
   * (and fills out the zeroes if A is not square)
   * @param A
   */
  def eyeizeFloat(A: CuMatrix[Float])(implicit handle: cublasHandle) {
    zeroOutFloat(A, 'U')
    zeroOutFloat(A, 'L')

    val diagLen = if (A.rows < A.cols) A.rows else A.cols
    val d_diag = CuMatrix.ones[Float](diagLen, 1)

    JCublas2.cublasScopy(handle, diagLen, d_diag.offsetPointer, 1, A.offsetPointer, A.majorStride + 1)
  }

  def eyeizeDouble(A: CuMatrix[Double])(implicit handle: cublasHandle) {
    zeroOutDouble(A, 'U')
    zeroOutDouble(A, 'L')

    val diagLen = if (A.rows < A.cols) A.rows else A.cols
    val d_diag = CuMatrix.fromDense(DenseMatrix.ones[Double](diagLen, 1))

    JCublas2.cublasDcopy(handle, diagLen, d_diag.offsetPointer, 1, A.offsetPointer, A.majorStride + 1)
  }

  /**
   * Performs an in-place transposition of the given part of the matrix A
   * If m and n are not A.rows and A.cols, it's only useful for m == n
   * @param m rows
   * @param n cols
   * @param A
   * @param Aroff row-wise offset
   * @param Acoff column-wise offset
   */
  def transposeInplaceFloat(m: Int, n: Int, A: CuMatrix[Float], Aroff: Int = 0, Acoff: Int = 0)(implicit handle: cublasHandle) {
    val B = CuMatrix.create[Float](A.rows, A.cols)

    val one = Pointer.pointerToFloat(1.0f).toCuPointer
    val zero = Pointer.pointerToFloat(0.0f).toCuPointer

    JCublas2.cublasSgeam(handle, cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, m, n,
      one, A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff) * A.elemSize), A.majorStride,
      zero, A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff) * A.elemSize), A.majorStride,
      B.offsetPointer.withByteOffset(B.linearIndex(Aroff, Acoff) * B.elemSize), B.majorStride)

    //copyFloat(n, m, A, Aroff, Acoff, B, Aroff, Acoff)
    A := B
  }

  def transposeInplaceDouble(m: Int, n: Int, A: CuMatrix[Double], Aroff: Int = 0, Acoff: Int = 0)(implicit handle: cublasHandle) {
    val B = CuMatrix.create[Double](A.rows, A.cols)
    val one = Pointer.pointerToDouble(1.0).toCuPointer
    val zero = Pointer.pointerToDouble(0.0).toCuPointer
    JCublas2.cublasDgeam(handle, cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, n, m,
      one, A.offsetPointer, A.majorStride,
      zero, A.offsetPointer, A.majorStride,
      B.offsetPointer, B.majorStride)

    A := B
  }

  def  transposeFloat(m: Int, n: Int, At: CuMatrix[Float], Atroff: Int, Atcoff: Int, A: CuMatrix[Float], Aroff: Int, Acoff: Int)(implicit handle: cublasHandle) {
    val one = Pointer.pointerToFloat(1.0f).toCuPointer
    val zero = Pointer.pointerToFloat(0.0f).toCuPointer
    JCublas2.cublasSgeam(handle, cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, m, n,
      one, A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff) * A.elemSize), A.majorStride,
      zero, A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff) * A.elemSize), A.majorStride,
      At.offsetPointer.withByteOffset(At.linearIndex(Atroff, Atcoff) * At.elemSize), At.majorStride)
  }

  def transposeDouble(m: Int, n: Int, At: CuMatrix[Double], Atroff: Int, Atcoff: Int, A: CuMatrix[Double], Aroff: Int, Acoff: Int)(implicit handle: cublasHandle) {
    val one = Pointer.pointerToDouble(1.0).toCuPointer
    val zero = Pointer.pointerToDouble(0.0).toCuPointer
    JCublas2.cublasDgeam(handle, cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, m, n,
      one, A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff) * A.elemSize), A.majorStride,
      zero, A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff) * A.elemSize), A.majorStride,
      At.offsetPointer.withByteOffset(At.linearIndex(Atroff, Atcoff) * At.elemSize), At.majorStride)
  }

  def batchSwapFloat(batchSize: Int, n: Int, ipiv: Array[Int], ipiv_off: Int, A: CuMatrix[Float], Aroff: Int, Acoff: Int)(implicit handle: cublasHandle) {
    if (n <= 0 || batchSize <= 0) return

    val SWAPS_PER_RUN = 64
    val VL = 64
    val nPtr = Pointer.pointerToInt(n).toCuPointer
    val ldaPtr = Pointer.pointerToInt(A.majorStride).toCuPointer

    // load the kernel:
    JCudaDriver.setExceptionsEnabled(true)
    implicit val dev = CuDevice(0)
    val ctx = CuContext.ensureContext
    val module = new CUmodule()
    val batchSwap = new CUfunction()
    JCudaDriver.cuModuleLoad(module, "src/main/resources/gust/linalg/cuda/swap.ptx")

    val funcName = "batch_sswap"
    JCudaDriver.cuModuleGetFunction(batchSwap, module, funcName)


    cfor(0)(_ < batchSize, _ + SWAPS_PER_RUN) { i => {
      val nb = if (SWAPS_PER_RUN < batchSize - i) SWAPS_PER_RUN else batchSize - i
      val cpu_ipiv = new DenseMatrix[Int](SWAPS_PER_RUN, 1)
      cfor(0)(_ < nb, _ + 1 ) { j => {
        cpu_ipiv(j, 0) = ipiv(i + j + ipiv_off) - i - 1
      }}

      val kernel_ipiv = CuMatrix.fromDense(cpu_ipiv)

      val params = jcuda.Pointer.to(
        Pointer.pointerToInt(nb).toCuPointer,
        nPtr,
        jcuda.Pointer.to(A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff + i) * A.elemSize)),
        ldaPtr,
        jcuda.Pointer.to(kernel_ipiv.offsetPointer)
      )

      JCudaDriver.cuLaunchKernel(batchSwap, (n+VL-1)/VL, 1, 1, VL, 1, 1, 0, null, params, null)
      JCudaDriver.cuCtxSynchronize()
    }}

  }

  /**
   * Reduction with multiplication, based on the code from JCuda.org/Samples/JCudaReduction,
   * which in turn is based on the reduction from NVidia 'reduction' sample.
   *
   * Copyright 1993-2010 NVIDIA Corporation.
   *
   * @param A matrix from which we extract a set of numbers to be reduced
   * @param Aroff row offset
   * @param Acoff column offset
   * @param lda stride (1 --> take a column, A.majorStride --> take a row, A.majorStride + 1 --> take a diagonal)
   * @param numElems
   * @return a product of all selected entries
   */
  def reduceMult[V](A: CuMatrix[V], Aroff: Int, Acoff: Int, lda: Int, numElems: Int)(implicit handle: cublasHandle, ct: ClassTag[V]): V = {

    val data = new CuMatrix[V](numElems, 1)
    val isOfTypeS = A.elemSize == 4
    val isOfTypeD = A.elemSize == 8
    if (!(isOfTypeS || isOfTypeD)) throw new UnsupportedOperationException("Can only handle matrices with elemSizes 4 and 8")

    val (moduleName, cublasOp) = if (isOfTypeS) ("src/main/resources/gust/linalg/cuda/reduceFloat.ptx", JCublas2.cublasScopy _)
                     else ("src/main/resources/gust/linalg/cuda/reduceDouble.ptx", JCublas2.cublasDcopy _)

    JCudaDriver.setExceptionsEnabled(true)
    implicit val dev = CuDevice(0)
    val ctx = CuContext.ensureContext
    val module = new CUmodule()
    val reduce = new CUfunction()
    JCudaDriver.cuModuleLoad(module, moduleName)

    JCudaDriver.cuModuleGetFunction(reduce, module, "reduce")

    // copy the appropriate part of the matrix into a (linear) array:
    cublasOp(handle, numElems, A.offsetPointer.withByteOffset(A.linearIndex(Aroff, Acoff) * A.elemSize),
      lda, data.offsetPointer, 1)

    val maxThreads = 256  // max. threads per block. adjusting this does affect performance
    val maxBlocks = 2048

    var nblocks = getBlockNum(numElems, maxBlocks, maxThreads)
    var nthreads = getThreadNum(numElems, maxBlocks, maxThreads)

    val idata = data
    val odata = CuMatrix.create[V](nblocks, 1)

    reduceIter[V](numElems, nthreads, nblocks, idata, odata, reduce)

    var s = nblocks
    while (s > 1) {
      nblocks = getBlockNum(s, maxBlocks, maxThreads)
      nthreads = getThreadNum(s, maxBlocks, maxThreads)

      reduceIter(s, nthreads, nblocks, odata, odata, reduce)
      s = (s + (nthreads*2-1)) / (nthreads*2)
    }


//    odata.data(0)
    val h_result = Pointer.allocateArray[V](odata.data.getIO, 1)

    JCuda.cudaMemcpy2D(h_result.toCuPointer, odata.elemSize,
      odata.offsetPointer,
      odata.majorStride * odata.elemSize, odata.elemSize, 1, cudaMemcpyKind.cudaMemcpyDeviceToHost)

    h_result(0)
  }

  /**
   * Performs a single iteration of the reduction
   */
  def reduceIter[V](size: Int, threads: Int, blocks: Int, idata: CuMatrix[V], odata: CuMatrix[V], func: CUfunction) {
    val sharedMemSize = threads * idata.elemSize.toInt * (if (threads <= 32) 2 else 1)

    val params = jcuda.Pointer.to(
      jcuda.Pointer.to(idata.offsetPointer),
      jcuda.Pointer.to(odata.offsetPointer),
      jcuda.Pointer.to(Array(size))
    )

    JCudaDriver.cuLaunchKernel(func, blocks, 1, 1,
      threads, 1, 1,
      sharedMemSize, null, params, null)
    JCudaDriver.cuCtxSynchronize()
  }

  private def getThreadNum(size: Int, maxBlocks: Int, maxThreads: Int): Int = {
    if (size < maxThreads*2) nextPow2((size+1) / 2) else maxThreads
  }

  private def getBlockNum(size: Int, maxBlocks: Int, maxThreads: Int): Int = {
    var blocks = 0
    val threads = getThreadNum(size, maxBlocks, maxThreads)
    blocks = (size + (threads*2 - 1)) / (threads*2)
    blocks = Math.min(maxBlocks, blocks)

    blocks
  }

  private def nextPow2(n: Int): Int = {
    var x = n - 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16

    x + 1
  }


  /* ------ methods for sparse matrices --------------------------------------------- */

  /**
   * Performs the addition: C = alpha * AS + beta * BS
   * @param alpha (scalar)
   * @param AS
   * @param beta  (scalar)
   * @param BS
   * @return C = alpha * A + beta * B
   */
  def sparseSgeam(alpha: Float, AS: CuSparseMatrix[Float], beta: Float, BS: CuSparseMatrix[Float])(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle): CuSparseMatrix[Float] = {

    val A = if (AS.isTranspose) AS.transpose else AS
    val B = if (BS.isTranspose) BS.transpose else BS

    require(A.rows == B.rows, s"Row dimension mismatch for addition: ${(A.rows, A.cols)} ${(B.rows, B.cols)}")
    require(A.cols == B.cols, s"Column dimension mismatch: ${(A.rows, A.cols)} ${(B.rows, B.cols)}")

    val m = A.cols
    val n = A.rows

    val nnzHost = Array(0)
    val nnzHostPtr = jcuda.Pointer.to(nnzHost)
    JCusparse2.cusparseSetPointerMode(sparseHandle, cusparsePointerMode.CUSPARSE_POINTER_MODE_HOST)

    val descrC = new cusparseMatDescr
    JCusparse2.cusparseCreateMatDescr(descrC)
    JCusparse2.cusparseSetMatType(descrC, cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL)
    JCusparse2.cusparseSetMatIndexBase(descrC, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO)
    JCusparse2.cusparseSetMatDiagType(descrC, cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT)

    val cscColPtrC = CuMatrix.create[Int](m+1, 1)

    JCusparse2.cusparseXcsrgeamNnz(sparseHandle, m, n, A.descr, A.nnz, A.cscColPtr.offsetPointer,
      A.cscRowInd.offsetPointer, B.descr, B.nnz, B.cscColPtr.offsetPointer, B.cscRowInd.offsetPointer,
      descrC, cscColPtrC.offsetPointer, nnzHostPtr)

    var nnzC = nnzHost(0)
    if (nnzC == 0) {  // trick from cusparse's doc page
      JCuda.cudaMemcpy(nnzHostPtr,
        cscColPtrC.offsetPointer.withByteOffset(cscColPtrC.linearIndex(m, 0) * cscColPtrC.elemSize),
        1, cudaMemcpyKind.cudaMemcpyDeviceToHost)
      nnzC = nnzHost(0)
    }

    val alphaPtr = jcuda.Pointer.to(Array(alpha))
    val betaPtr = jcuda.Pointer.to(Array(beta))

    val cscValC = CuMatrix.create[Float](nnzC, 1)
    val cscRowIndC = CuMatrix.create[Int](nnzC, 1)

    JCusparse2.cusparseScsrgeam(sparseHandle, m, n, alphaPtr, A.descr, A.nnz, A.cscVal.offsetPointer,
      A.cscColPtr.offsetPointer, A.cscRowInd.offsetPointer, betaPtr, B.descr, B.nnz, B.cscVal.offsetPointer,
      B.cscColPtr.offsetPointer, B.cscRowInd.offsetPointer, descrC, cscValC.offsetPointer, cscColPtrC.offsetPointer, cscRowIndC.offsetPointer)

    new CuSparseMatrix[Float](

      n, m, descrC, cscValC, cscColPtrC, cscRowIndC)
  }

  def sparseDgeam(alpha: Double, AS: CuSparseMatrix[Double], beta: Double, BS: CuSparseMatrix[Double])(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle): CuSparseMatrix[Double] = {

    val A = if (AS.isTranspose) AS.transpose else AS
    val B = if (BS.isTranspose) BS.transpose else BS

    require(A.rows == B.rows, s"Row dimension mismatch for addition: ${(A.rows, A.cols)} ${(B.rows, B.cols)}")
    require(A.cols == B.cols, s"Column dimension mismatch: ${(A.rows, A.cols)} ${(B.rows, B.cols)}")

    val m = A.cols
    val n = A.rows

    val nnzHost = Array(0)
    val nnzHostPtr = jcuda.Pointer.to(nnzHost)
    JCusparse2.cusparseSetPointerMode(sparseHandle, cusparsePointerMode.CUSPARSE_POINTER_MODE_HOST)

    val descrC = new cusparseMatDescr
    JCusparse2.cusparseCreateMatDescr(descrC)
    JCusparse2.cusparseSetMatType(descrC, cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL)
    JCusparse2.cusparseSetMatIndexBase(descrC, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO)
    JCusparse2.cusparseSetMatDiagType(descrC, cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT)

    val cscColPtrC = CuMatrix.create[Int](m+1, 1)

    JCusparse2.cusparseXcsrgeamNnz(sparseHandle, m, n, A.descr, A.nnz, A.cscColPtr.offsetPointer,
      A.cscRowInd.offsetPointer, B.descr, B.nnz, B.cscColPtr.offsetPointer, B.cscRowInd.offsetPointer,
      descrC, cscColPtrC.offsetPointer, nnzHostPtr)

    var nnzC = nnzHost(0)
    if (nnzC == 0) {  // trick from cusparse's doc page
      JCuda.cudaMemcpy(nnzHostPtr,
        cscColPtrC.offsetPointer.withByteOffset(cscColPtrC.linearIndex(m, 0) * cscColPtrC.elemSize),
        1, cudaMemcpyKind.cudaMemcpyDeviceToHost)
      nnzC = nnzHost(0)
    }

    val alphaPtr = jcuda.Pointer.to(Array(alpha))
    val betaPtr = jcuda.Pointer.to(Array(beta))

    val cscValC = CuMatrix.create[Double](nnzC, 1)
    val cscRowIndC = CuMatrix.create[Int](nnzC, 1)

    JCusparse2.cusparseDcsrgeam(sparseHandle, m, n, alphaPtr, A.descr, A.nnz, A.cscVal.offsetPointer,
      A.cscColPtr.offsetPointer, A.cscRowInd.offsetPointer, betaPtr, B.descr, B.nnz, B.cscVal.offsetPointer,
      B.cscColPtr.offsetPointer, B.cscRowInd.offsetPointer, descrC, cscValC.offsetPointer, cscColPtrC.offsetPointer, cscRowIndC.offsetPointer)

    new CuSparseMatrix[Double](n, m, descrC, cscValC, cscColPtrC, cscRowIndC)
  }

  /**
   * Computes matrix product: C = A * B
   * @param A
   * @param B
   * @return C
   */
  def sparseSgemm(A: CuSparseMatrix[Float], B: CuSparseMatrix[Float])(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle) = {
    val transA = A.isTranspose
    val transB = B.isTranspose

    val m = if (transA) A.cols else A.rows
    val n = if (transB) B.rows else B.cols

    val k = if (transA) A.rows else A.cols
    val k_ = if (transB) B.cols else B.rows

    require(k == k_, s"Dimension mismatch for multiplication: ${(A.rows, A.cols)} ${(B.rows, B.cols)}")

    val nnzHost = Array(0)
    val nnzHostPtr = jcuda.Pointer.to(nnzHost)
    JCusparse2.cusparseSetPointerMode(sparseHandle, cusparsePointerMode.CUSPARSE_POINTER_MODE_HOST)

    val descrC = new cusparseMatDescr
    JCusparse2.cusparseCreateMatDescr(descrC)
    JCusparse2.cusparseSetMatType(descrC, cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL)
    JCusparse2.cusparseSetMatIndexBase(descrC, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO)
    JCusparse2.cusparseSetMatDiagType(descrC, cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT)

    val csrRowPtrC = CuMatrix.create[Int](m+1, 1)
    val opA = if (transA) cusparseOperation.CUSPARSE_OPERATION_TRANSPOSE else cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE
    val opB = if (transB) cusparseOperation.CUSPARSE_OPERATION_TRANSPOSE else cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE

    JCusparse2.cusparseXcsrgemmNnz(sparseHandle, opA, opB, m, n, k, B.descr, B.nnz, B.cscColPtr.offsetPointer,
      B.cscRowInd.offsetPointer, A.descr, A.nnz, A.cscColPtr.offsetPointer, A.cscRowInd.offsetPointer,
      descrC, csrRowPtrC.offsetPointer, nnzHostPtr)

    var nnzC = nnzHost(0)
    if (nnzC == 0) {
      JCuda.cudaMemcpy(nnzHostPtr,
        csrRowPtrC.offsetPointer.withByteOffset(csrRowPtrC.linearIndex(m, 0) * csrRowPtrC.elemSize),
        1, cudaMemcpyKind.cudaMemcpyDeviceToHost)
      nnzC = nnzHost(0)
    }

    val csrValC = CuMatrix.create[Float](nnzC, 1)
    val csrColIndC = CuMatrix.create[Int](nnzC, 1)

    JCusparse2.cusparseScsrgemm(sparseHandle, opA, opB, m, n, k, B.descr, B.nnz, B.cscVal.offsetPointer,
      B.cscColPtr.offsetPointer, B.cscRowInd.offsetPointer, A.descr, A.nnz, A.cscVal.offsetPointer,
      A.cscColPtr.offsetPointer, A.cscRowInd.offsetPointer, descrC, csrValC.offsetPointer, csrRowPtrC.offsetPointer, csrColIndC.offsetPointer)

    new CuSparseMatrix[Float](m, n, descrC, csrValC, csrRowPtrC, csrColIndC)
  }

  def sparseDgemm(A: CuSparseMatrix[Double], B: CuSparseMatrix[Double])(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle) = {
    val transA = A.isTranspose
    val transB = B.isTranspose

    val m = if (transA) A.cols else A.rows
    val n = if (transB) B.rows else B.cols

    val k = if (transA) A.rows else A.cols
    val k_ = if (transB) B.cols else B.rows

    require(k == k_, s"Dimension mismatch for multiplication: ${(A.rows, A.cols)} ${(B.rows, B.cols)}")

    val nnzHost = Array(0)
    val nnzHostPtr = jcuda.Pointer.to(nnzHost)
    JCusparse2.cusparseSetPointerMode(sparseHandle, cusparsePointerMode.CUSPARSE_POINTER_MODE_HOST)

    val descrC = new cusparseMatDescr
    JCusparse2.cusparseCreateMatDescr(descrC)
    JCusparse2.cusparseSetMatType(descrC, cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL)
    JCusparse2.cusparseSetMatIndexBase(descrC, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO)
    JCusparse2.cusparseSetMatDiagType(descrC, cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT)

    val csrRowPtrC = CuMatrix.create[Int](m+1, 1)
    val opA = if (transA) cusparseOperation.CUSPARSE_OPERATION_TRANSPOSE else cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE
    val opB = if (transB) cusparseOperation.CUSPARSE_OPERATION_TRANSPOSE else cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE

    JCusparse2.cusparseXcsrgemmNnz(sparseHandle, opA, opB, m, n, k, B.descr, B.nnz, B.cscColPtr.offsetPointer,
      B.cscRowInd.offsetPointer, A.descr, A.nnz, A.cscColPtr.offsetPointer, A.cscRowInd.offsetPointer,
      descrC, csrRowPtrC.offsetPointer, nnzHostPtr)

    var nnzC = nnzHost(0)
    if (nnzC == 0) {
      JCuda.cudaMemcpy(nnzHostPtr,
        csrRowPtrC.offsetPointer.withByteOffset(csrRowPtrC.linearIndex(m, 0) * csrRowPtrC.elemSize),
        1, cudaMemcpyKind.cudaMemcpyDeviceToHost)
      nnzC = nnzHost(0)
    }

    val csrValC = CuMatrix.create[Double](nnzC, 1)
    val csrColIndC = CuMatrix.create[Int](nnzC, 1)

    JCusparse2.cusparseDcsrgemm(sparseHandle, opA, opB, m, n, k, B.descr, B.nnz, B.cscVal.offsetPointer,
      B.cscColPtr.offsetPointer, B.cscRowInd.offsetPointer, A.descr, A.nnz, A.cscVal.offsetPointer,
      A.cscColPtr.offsetPointer, A.cscRowInd.offsetPointer, descrC, csrValC.offsetPointer, csrRowPtrC.offsetPointer, csrColIndC.offsetPointer)

    new CuSparseMatrix[Double](m, n, descrC, csrValC, csrRowPtrC, csrColIndC)
  }

  /**
   * Performs the sparse_matrix * dense_vector multiplication (resulting in a dense vector)
   * @param AS sparse matrix
   * @param b dense vector
   * @return AS * b, dense vector
   */
  def sparseSgemv(AS: CuSparseMatrix[Float], b: CuMatrix[Float])(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle) = {
    // we cannot do a vector*matrix instead of matrix*vector, so we have to transpose:
    val transA = if (AS.isTranspose) cusparseOperation.CUSPARSE_OPERATION_TRANSPOSE else cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE
    val A = AS.transpose

    val m = AS.rows
    val n = AS.cols // rows and cols of the original matrix (without the transpose)

    require(b.cols == 1, "b must be a vector")
    require(A.cols == b.rows, s"Dimension mismatch: A.rows = ${A.rows}, b.length = ${b.cols}")

    val res = CuMatrix.create[Float](m, 1)
    val zero = jcuda.Pointer.to(Array(0.0f))
    val one = jcuda.Pointer.to(Array(1.0f))

    JCusparse2.cusparseScsrmv(sparseHandle, transA, m, n, A.nnz, one,
      A.descr, A.cscVal.offsetPointer, A.cscColPtr.offsetPointer, A.cscRowInd.offsetPointer,
      b.offsetPointer, zero, res.offsetPointer)

    res
  }

  def sparseDgemv(A: CuSparseMatrix[Double], b: CuMatrix[Double])(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle) = {
    // we cannot do a vector*matrix instead of matrix*vector, so we have to transpose:
    // cusparse is CSR-based, while CuSparseMatrix is CSC.
    val transA = if (!A.isTranspose) cusparseOperation.CUSPARSE_OPERATION_TRANSPOSE else cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE

    val _A = if(A.isTranspose) A else A.t

    val m = _A.rows
    val n = _A.cols // rows and cols of the original matrix (without the transpose)

    require(b.cols == 1, "b must be a vector")
    require(A.cols == b.rows, s"Dimension mismatch: A.cols = ${A.cols}, b.length = ${b.rows}")

    val res = CuMatrix.create[Double](m, 1)
    val zero = jcuda.Pointer.to(Array(0.0))
    val one = jcuda.Pointer.to(Array(1.0))

    JCusparse2.cusparseDcsrmv(sparseHandle, transA, m, n, A.nnz, one,
      _A.descr, _A.cscVal.offsetPointer, _A.cscColPtr.offsetPointer, _A.cscRowInd.offsetPointer,
      b.offsetPointer, zero, res.offsetPointer)

    res
  }

}
