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
   * Hmm, there's a cublas function for the addition...
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

  def zeroOutFloatOffset(A: CuMatrix[Float], Aroff: Int, Acoff: Int, fillMode: Char, includeDiagonal: Boolean = false) {
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

  def SgemmTN(m: Int, n: Int, k: Int, alpha: jcuda.Pointer,
              A: CuMatrix[Float], Aroff: Int, Acoff: Int,
              B: CuMatrix[Float], Broff: Int, Bcoff: Int,
              beta: jcuda.Pointer,
              C: CuMatrix[Float], Croff: Int, Ccoff: Int)(implicit handle: cublasHandle) {

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
              C: CuMatrix[Double], Croff: Int, Ccoff: Int)(implicit handle: cublasHandle) {

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
              C: CuMatrix[Float], Croff: Int, Ccoff: Int)(implicit handle: cublasHandle) {

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
              C: CuMatrix[Double], Croff: Int, Ccoff: Int)(implicit handle: cublasHandle) {

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


  /*
   * Functions for calculating the residuals:
   * calculates |A - BC| where BC are Q and R in case of QR factorization or L and U
   * in case of LU factorization.
   * If the pivot matrix is not null, we actually calculate |PA - BC|.
   *
   * It can also calculate the residual in case of the solve method, since we treat vectors as matrices
   */
  def residualFloat(A: CuMatrix[Float], B: CuMatrix[Float], C: CuMatrix[Float], P: CuMatrix[Float] = null): Double = {
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

    implicit val handle = A.blas

    val d_A = CuMatrix.create[Float](A.rows, A.cols)

    val minusOneArr = Array(-1.0f)
    val oneArr = Array(1.0f)
    val zeroArr = Array(0.0f)

    if (P != null)
      // d_A = P*A
      SgemmNN(A.rows, A.cols, A.cols, jcuda.Pointer.to(oneArr), P, 0, 0, A, 0, 0, jcuda.Pointer.to(zeroArr), d_A, 0, 0)
    else
      d_A := A

    // d_A = d_A - B*C
    SgemmNN(d_A.rows, d_A.cols, C.cols, jcuda.Pointer.to(minusOneArr), B, 0, 0, C, 0, 0, jcuda.Pointer.to(oneArr), d_A, 0, 0)

    d_A.norm
  }

  def residualDouble(A: CuMatrix[Double], B: CuMatrix[Double], C: CuMatrix[Double], P: CuMatrix[Double] = null): Double = {
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

    implicit val handle = A.blas

    val d_A = CuMatrix.create[Double](A.rows, A.cols)

    val minusOneArr = Array(-1.0)
    val oneArr = Array(1.0)
    val zeroArr = Array(0.0)

    if (P != null)
    // d_A = P*A
      DgemmNN(A.rows, A.cols, A.cols, jcuda.Pointer.to(oneArr), P, 0, 0, A, 0, 0, jcuda.Pointer.to(zeroArr), d_A, 0, 0)
    else
      d_A := A

    // d_A = d_A - B*C
    DgemmNN(d_A.rows, d_A.cols, C.cols, jcuda.Pointer.to(minusOneArr), B, 0, 0, C, 0, 0, jcuda.Pointer.to(oneArr), d_A, 0, 0)

    d_A.norm
  }

  /**
   * Overwrites matrix A with an identity matrix
   * (and fills out the zeroes if A is not square)
   * @param A
   */
  def eyeizeFloat(A: CuMatrix[Float]) {
    implicit val handle = A.blas

    zeroOutFloat(A, 'U')
    zeroOutFloat(A, 'L')

    val diagLen = if (A.rows < A.cols) A.rows else A.cols
    val d_diag = CuMatrix.ones[Float](diagLen, 1)

    JCublas2.cublasScopy(handle, diagLen, d_diag.offsetPointer, 1, A.offsetPointer, A.majorStride + 1)
  }

  def eyeizeDouble(A: CuMatrix[Double]) {
    implicit val handle = A.blas

    zeroOutDouble(A, 'U')
    zeroOutDouble(A, 'L')

    val diagLen = if (A.rows < A.cols) A.rows else A.cols
    val d_diag = CuMatrix.fromDense(DenseMatrix.ones[Double](diagLen, 1))

    JCublas2.cublasDcopy(handle, diagLen, d_diag.offsetPointer, 1, A.offsetPointer, A.majorStride + 1)
  }
}
