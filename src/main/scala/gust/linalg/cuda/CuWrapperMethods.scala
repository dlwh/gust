package gust.linalg.cuda

import jcuda.jcublas.{cublasOperation, JCublas2, cublasHandle}
import breeze.linalg.DenseMatrix
import jcuda.runtime.{cudaMemcpyKind, JCuda}
import jcuda.driver.{CUfunction, CUmodule, JCudaDriver}
import gust.util.cuda.{CuContext, CuDevice}


/**
 * Created by piotrek on 16.06.2014.
 *
 */
object CuWrapperMethods {

  /*
   * Kernels for element-wise operations: product and sum
   * They may be moved to CuMatrix later
   */
  def elemWiseProdFloat(A: CuMatrix[Float], B: CuMatrix[Float]): CuMatrix[Float] = elemWiseFloat('p', A, B)

  def elemWiseProdDouble(A: CuMatrix[Double], B: CuMatrix[Double]): CuMatrix[Double] = elemWiseDouble('p', A, B)

  def elemWiseSumFloat(A: CuMatrix[Float], B: CuMatrix[Float]): CuMatrix[Float] = elemWiseFloat('s', A, B)

  def elemWiseSumDouble(A: CuMatrix[Double], B: CuMatrix[Double]): CuMatrix[Double] = elemWiseDouble('s', A, B)


  private def elemWiseFloat(operation: Char, A: CuMatrix[Float], B: CuMatrix[Float]): CuMatrix[Float] = {
    if (A.rows != B.rows || A.cols != B.cols) {
      println("Matrices have to be of the same dimensions")
      return null
    }

    implicit val handle: cublasHandle = A.blas
    val C = CuMatrix.create[Float](A.rows, A.cols)

    JCudaDriver.setExceptionsEnabled(true)
    implicit val dev = CuDevice(0)
    val ctx = CuContext.ensureContext
    val module = new CUmodule()
    val function = new CUfunction()
    val func_name = if (operation == 'p') "hadamard" else "matrix_sum"
    JCudaDriver.cuModuleLoad(module, "src/main/resources/gust/linalg/cuda/elemWiseFloat.ptx")
    JCudaDriver.cuModuleGetFunction(function, module, func_name)

    // kernel parameters:
    val ldaArr = Array(A.majorStride, B.majorStride, C.majorStride)
    val lda = jcuda.Pointer.to(ldaArr)
    val ldb = jcuda.Pointer.to(ldaArr).withByteOffset(jcuda.Sizeof.INT)
    val ldc = jcuda.Pointer.to(ldaArr).withByteOffset(jcuda.Sizeof.INT * 2)
    val dimsArr = Array(A.rows, A.cols)
    val m = jcuda.Pointer.to(dimsArr)
    val n = jcuda.Pointer.to(dimsArr).withByteOffset(jcuda.Sizeof.INT)

    val params = jcuda.Pointer.to(
      m, n,
      jcuda.Pointer.to(A.offsetPointer), lda,
      jcuda.Pointer.to(B.offsetPointer), ldb,
      jcuda.Pointer.to(C.offsetPointer), ldc
    )

    val nb = 32
    val gridDim = (A.rows / nb + (if (A.rows % nb == 0) 0 else 1),
      A.cols / nb + (if (A.cols % nb == 0) 0 else 1),
      1)
    val blockDim = (nb, nb, 1)

    JCudaDriver.cuLaunchKernel(function, gridDim._1, gridDim._2, gridDim._3,
      blockDim._1, blockDim._2, blockDim._3,
      0, null, params, null)
    JCudaDriver.cuCtxSynchronize()

    C
  }

  private def elemWiseDouble(operation: Char, A: CuMatrix[Double], B: CuMatrix[Double]): CuMatrix[Double] = {
    if (A.rows != B.rows || A.cols != B.cols) {
      println("Matrices have to be of the same dimensions")
      return null
    }

    implicit val handle: cublasHandle = A.blas
    val C = CuMatrix.create[Double](A.rows, A.cols)

    JCudaDriver.setExceptionsEnabled(true)
    implicit val dev = CuDevice(0)
    val ctx = CuContext.ensureContext
    val module = new CUmodule()
    val function = new CUfunction()
    val func_name = if (operation == 'p') "hadamard" else "matrix_sum"
    JCudaDriver.cuModuleLoad(module, "src/main/resources/gust/linalg/cuda/elemWiseDouble.ptx")
    JCudaDriver.cuModuleGetFunction(function, module, func_name)

    // kernel parameters:
    val ldaArr = Array(A.majorStride, B.majorStride, C.majorStride)
    val lda = jcuda.Pointer.to(ldaArr)
    val ldb = jcuda.Pointer.to(ldaArr).withByteOffset(jcuda.Sizeof.INT)
    val ldc = jcuda.Pointer.to(ldaArr).withByteOffset(jcuda.Sizeof.INT * 2)
    val dimsArr = Array(A.rows, A.cols)
    val m = jcuda.Pointer.to(dimsArr)
    val n = jcuda.Pointer.to(dimsArr).withByteOffset(jcuda.Sizeof.INT)

    val params = jcuda.Pointer.to(
      m, n,
      jcuda.Pointer.to(A.offsetPointer), lda,
      jcuda.Pointer.to(B.offsetPointer), ldb,
      jcuda.Pointer.to(C.offsetPointer), ldc
    )

    val nb = 32
    val gridDim = (A.rows / nb + (if (A.rows % nb == 0) 0 else 1),
      A.cols / nb + (if (A.cols % nb == 0) 0 else 1),
      1)
    val blockDim = (nb, nb, 1)

    JCudaDriver.cuLaunchKernel(function, gridDim._1, gridDim._2, gridDim._3,
      blockDim._1, blockDim._2, blockDim._3,
      0, null, params, null)
    JCudaDriver.cuCtxSynchronize()

    C
  }



  /*
   * wrapped calls to kernels for zeroing out some parts of matrices.
   */
  def zeroOutFloat(A: CuMatrix[Float], fillMode: Char, includeDiagonal: Boolean = false) {
    val nb = 32     // most cards can go as high as 1024 (32**2) threads per block

    JCudaDriver.setExceptionsEnabled(true)
    implicit val dev = CuDevice(0)
    val ctx = CuContext.ensureContext
    val module = new CUmodule()
    val zero_out = new CUfunction()
    JCudaDriver.cuModuleLoad(module, "src/main/resources/gust/linalg/cuda/enforceLUFloat.ptx")

    // once again -- magnled names, I'll have try to figure something out
    val funcName = if (fillMode == 'U') "_Z6zerosUiiPfii" else "_Z6zerosLiiPfii"
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
    val funcName = if (fillMode == 'U') "_Z6zerosUiiPdii" else "_Z6zerosLiiPdii"
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

  /*
   * wrapped calls to gemm, don't require passing lda's and elemSizes around
   */
  def SgemmNN(m: Int, n: Int, k: Int, alpha: jcuda.Pointer,
              A: CuMatrix[Float], Aroff: Int, Acoff: Int,
              B: CuMatrix[Float], Broff: Int, Bcoff: Int,
              beta: jcuda.Pointer,
              C: CuMatrix[Float], Croff: Int, Ccoff: Int)(implicit handle: cublasHandle) {

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
              C: CuMatrix[Double], Croff: Int, Ccoff: Int)(implicit handle: cublasHandle) {

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
              C: CuMatrix[Float], Croff: Int, Ccoff: Int)(implicit handle: cublasHandle) {

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
              C: CuMatrix[Double], Croff: Int, Ccoff: Int)(implicit handle: cublasHandle) {

    JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T,
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
                  src: DenseMatrix[Float], src_roff: Int, src_coff: Int) {
    JCuda.cudaMemcpy2D(dst.offsetPointer.withByteOffset(dst.linearIndex(dst_roff, dst_coff) * dst.elemSize),
      dst.majorStride * dst.elemSize,
      jcuda.Pointer.to(src.data).withByteOffset(src.linearIndex(src_roff, src_coff) * dst.elemSize),
      src.majorStride * dst.elemSize, m * dst.elemSize, n,
      cudaMemcpyKind.cudaMemcpyHostToDevice)
  }

  def uploadDouble(m: Int, n: Int, dst: CuMatrix[Double],
                   dst_roff: Int, dst_coff: Int,
                   src: DenseMatrix[Double], src_roff: Int, src_coff: Int) {
    JCuda.cudaMemcpy2D(dst.offsetPointer.withByteOffset(dst.linearIndex(dst_roff, dst_coff) * dst.elemSize),
      dst.majorStride * dst.elemSize,
      jcuda.Pointer.to(src.data).withByteOffset(src.linearIndex(src_roff, src_coff) * dst.elemSize),
      src.majorStride * dst.elemSize, m * dst.elemSize, n,
      cudaMemcpyKind.cudaMemcpyHostToDevice)
  }

  def downloadFloat(m: Int, n: Int, dst: DenseMatrix[Float], dst_roff: Int, dst_coff: Int,
                    src: CuMatrix[Float], src_roff: Int, src_coff: Int) {
    JCuda.cudaMemcpy2D(jcuda.Pointer.to(dst.data).withByteOffset(dst.linearIndex(dst_roff, dst_coff) * src.elemSize),
      dst.majorStride * src.elemSize,
      src.offsetPointer.withByteOffset(src.linearIndex(src_roff, src_coff) * src.elemSize),
      src.majorStride * src.elemSize, m * src.elemSize, n, cudaMemcpyKind.cudaMemcpyDeviceToHost)
  }

  def downloadDouble(m: Int, n: Int, dst: DenseMatrix[Double], dst_roff: Int, dst_coff: Int,
                     src: CuMatrix[Double], src_roff: Int, src_coff: Int) {
    JCuda.cudaMemcpy2D(jcuda.Pointer.to(dst.data).withByteOffset(dst.linearIndex(dst_roff, dst_coff) * src.elemSize),
      dst.majorStride * src.elemSize,
      src.offsetPointer.withByteOffset(src.linearIndex(src_roff, src_coff) * src.elemSize),
      src.majorStride * src.elemSize, m * src.elemSize, n, cudaMemcpyKind.cudaMemcpyDeviceToHost)
  }

}
