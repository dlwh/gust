package gust.linalg.cuda

import jcuda.jcublas.cublasHandle
import breeze.linalg.{DenseMatrix, DenseVector}
import org.netlib.util.intW
import jcuda.driver.{CUfunction, CUmodule, JCudaDriver}
import gust.util.cuda.{CuContext, CuDevice}
import spire.syntax.cfor._
import gust.linalg.cuda.CuWrapperMethods._
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}

/**
 * Created by piotrek on 28.06.2014.
 */
object CuQR {

  /**
   * QR factorization for matrices of size m x n (where m >= n)
   * @param A
   * @param handle
   * @return
   */
  def QRFloatMN(A: CuMatrix[Float])(implicit handle: cublasHandle): (CuMatrix[Float], DenseVector[Float]) = {
    if (A.rows < A.cols) {
      println("Number of rows of matrix A cannot be smaller than the number of columns.")
      return (A, null)
    }

    // pointers to scalars (for dgemm):
    val oneArr = Array(1.0f)
    val hostOne = jcuda.Pointer.to(oneArr)
    val zeroArr = Array(0.0f)
    val hostZero = jcuda.Pointer.to(zeroArr)
    val minusOneArr = Array(-1.0f)
    val hostMinusOne = jcuda.Pointer.to(minusOneArr)


    val nb = if (A.cols < 64) A.cols else 64
    val ldaArr = Array(A.majorStride)
    val lda = jcuda.Pointer.to(ldaArr)
    val m = A.rows
    val n = A.cols

    // gpu matrices:
    val gpu_matrix = CuMatrix.create[Float](m, n);
    gpu_matrix := A
    val gpu_TV = CuMatrix.create[Float](nb, m)
    val gpu_TVA = CuMatrix.create[Float](nb, m)

    // cpu matrices:
    val cpu_matrix = A.toDense
    val cpu_tau = DenseVector.zeros[Float](n)
    val cpu_work = Array.ofDim[Float](m * n * nb)
    val cpu_T = DenseMatrix.zeros[Float](nb, nb)

    var h, w = 0
    val es = gpu_matrix.elemSize
    val info = new intW(0)
    val lwork = cpu_work.length


    // prep for launching the kernel:
    JCudaDriver.setExceptionsEnabled(true)
    implicit val dev = CuDevice(0)
    val ctx = CuContext.ensureContext
    val module = new CUmodule()
    JCudaDriver.cuModuleLoad(module, "src/main/resources/gust/linalg/cuda/enforceLUFloat.ptx")

    val enfLU = new CUfunction()
    JCudaDriver.cuModuleGetFunction(enfLU, module, "_Z9enforceLUPfi")

    // we don't need to upload anything onto the GPU -- it's already there
    // not really sure about the 'm' here
    cfor(0)(_ < n, _ + nb) { i => {
      h = m - i
      w = if (h < nb) h else nb

      if (i > 0) {

        SgemmNN(nb, w, h + nb, hostOne, gpu_TV, 0, 0, gpu_matrix, i - nb, i, hostZero,
          gpu_TVA, 0, 0)

        SgemmNN(h + nb, w, nb, hostMinusOne, gpu_matrix, i - nb, i - nb, gpu_TVA, 0, 0, hostOne,
          gpu_matrix, i - nb, i)

        downloadFloat(m, w, cpu_matrix, 0, i, gpu_matrix, 0, i)

        SgemmNN(nb, h - nb, h + nb, hostOne, gpu_TV, 0, 0, gpu_matrix, i - nb, i + nb, hostZero,
          gpu_TVA, 0, 0)

        SgemmNN(h + nb, h - nb, nb, hostMinusOne, gpu_matrix, i - nb, i - nb, gpu_TVA, 0, 0, hostOne,
          gpu_matrix, i - nb, i + nb)

      }

      // factorization on CPU
      // additional params after matrices are offsets (like i in float *A;  A+i)
      lapack.sgeqrf(h, w, cpu_matrix.data, cpu_matrix.linearIndex(i, i), cpu_matrix.majorStride,
        cpu_tau.data, i, cpu_work, 0, lwork, info)


      if (h > nb) {
        lapack.slarft("F", "C", h, w, cpu_matrix.data, cpu_matrix.linearIndex(i, i), cpu_matrix.majorStride,
          cpu_tau.data, i, cpu_T.data, 0, cpu_T.majorStride)

        // transpose cpu_T:
        cfor(0)(_ < nb, _ + 1) { j => {
          cfor(0)(_ < j, _ + 1) { k => {
            cpu_T(j, k) = cpu_T(k, j)
            cpu_T(k, j) = 0.0f
          }
          }
        }
        }


        // upload to GPU:
        uploadFloat(nb, nb, gpu_TVA, 0, 0, cpu_T, 0, 0)
        uploadFloat(h, nb, gpu_matrix, i, i, cpu_matrix, i, i)

        // enforceLU, kernel launch:
        val params = jcuda.Pointer.to(
          jcuda.Pointer.to(gpu_matrix.offsetPointer.withByteOffset(gpu_matrix.linearIndex(i, i) * es)),
          lda
        )

        JCudaDriver.cuLaunchKernel(enfLU, nb, 1, 1, nb, 1, 1, 0, null, params, null)
        JCudaDriver.cuCtxSynchronize()

        SgemmNT(nb, h, nb, hostOne, gpu_TVA, 0, 0, gpu_matrix, i, i, hostZero, gpu_TV, 0, 0)
      }
    }
    }

    (CuMatrix.fromDense(cpu_matrix), cpu_tau)
  }


  /**
   * QR factorization for matrices of size m x n (where m >= n)
   * @param A
   * @param handle
   * @return
   */
  def QRDoubleMN(A: CuMatrix[Double])(implicit handle: cublasHandle): (CuMatrix[Double], DenseVector[Double]) = {
    if (A.rows < A.cols) {
      println("Number of rows of matrix A cannot be smaller than the number of columns.")
      return (A, null)
    }

    // pointers to scalars (for dgemm):
    val oneArr = Array(1.0)
    val hostOne = jcuda.Pointer.to(oneArr)
    val zeroArr = Array(0.0)
    val hostZero = jcuda.Pointer.to(zeroArr)
    val minusOneArr = Array(-1.0)
    val hostMinusOne = jcuda.Pointer.to(minusOneArr)


    val nb = if (A.cols < 64) A.cols else 64
    val ldaArr = Array(A.majorStride)
    val lda = jcuda.Pointer.to(ldaArr)
    val m = A.rows
    val n = A.cols

    // gpu matrices:
    val gpu_matrix = CuMatrix.create[Double](m, n);
    gpu_matrix := A
    val gpu_TV = CuMatrix.create[Double](nb, m)
    val gpu_TVA = CuMatrix.create[Double](nb, m)

    // cpu matrices:
    val cpu_matrix = A.toDense
    val cpu_tau = DenseVector.zeros[Double](n)
    val cpu_work = Array.ofDim[Double](m * n * nb)
    val cpu_T = DenseMatrix.zeros[Double](nb, nb)

    var h, w = 0
    val es = gpu_matrix.elemSize
    val info = new intW(0)
    val lwork = cpu_work.length


    // prep for launching the kernel:
    JCudaDriver.setExceptionsEnabled(true)
    implicit val dev = CuDevice(0)
    val ctx = CuContext.ensureContext
    val module = new CUmodule()
    JCudaDriver.cuModuleLoad(module, "src/main/resources/gust/linalg/cuda/enforceLUDouble.ptx")

    val enfLU = new CUfunction()
    JCudaDriver.cuModuleGetFunction(enfLU, module, "_Z9enforceLUPdi")

    // we don't need to upload anything onto the GPU -- it's already there
    // not really sure about the 'm' here
    cfor(0)(_ < n, _ + nb) { i => {
      h = m - i
      w = if (h < nb) h else nb

      if (i > 0) {

        DgemmNN(nb, w, h + nb, hostOne, gpu_TV, 0, 0, gpu_matrix, i - nb, i, hostZero,
          gpu_TVA, 0, 0)

        DgemmNN(h + nb, w, nb, hostMinusOne, gpu_matrix, i - nb, i - nb, gpu_TVA, 0, 0, hostOne,
          gpu_matrix, i - nb, i)

        downloadDouble(m, w, cpu_matrix, 0, i, gpu_matrix, 0, i)

        DgemmNN(nb, h - nb, h + nb, hostOne, gpu_TV, 0, 0, gpu_matrix, i - nb, i + nb, hostZero,
          gpu_TVA, 0, 0)

        DgemmNN(h + nb, h - nb, nb, hostMinusOne, gpu_matrix, i - nb, i - nb, gpu_TVA, 0, 0, hostOne,
          gpu_matrix, i - nb, i + nb)

      }

      // factorization on CPU
      // additional params after matrices are offsets (like i in float *A;  A+i)
      lapack.dgeqrf(h, w, cpu_matrix.data, cpu_matrix.linearIndex(i, i), cpu_matrix.majorStride,
        cpu_tau.data, i, cpu_work, 0, lwork, info)


      if (h > nb) {
        lapack.dlarft("F", "C", h, w, cpu_matrix.data, cpu_matrix.linearIndex(i, i), cpu_matrix.majorStride,
          cpu_tau.data, i, cpu_T.data, 0, cpu_T.majorStride)

        // transpose cpu_T:
        cfor(0)(_ < nb, _ + 1) { j => {
          cfor(0)(_ < j, _ + 1) { k => {
            cpu_T(j, k) = cpu_T(k, j)
            cpu_T(k, j) = 0.0
          }
          }
        }
        }


        // upload to GPU:
        uploadDouble(nb, nb, gpu_TVA, 0, 0, cpu_T, 0, 0)
        uploadDouble(h, nb, gpu_matrix, i, i, cpu_matrix, i, i)

        // enforceLU, kernel launch:
        val params = jcuda.Pointer.to(
          jcuda.Pointer.to(gpu_matrix.offsetPointer.withByteOffset(gpu_matrix.linearIndex(i, i) * es)),
          lda
        )

        JCudaDriver.cuLaunchKernel(enfLU, nb, 1, 1, nb, 1, 1, 0, null, params, null)
        JCudaDriver.cuCtxSynchronize()

        DgemmNT(nb, h, nb, hostOne, gpu_TVA, 0, 0, gpu_matrix, i, i, hostZero, gpu_TV, 0, 0)
      }
    }
    }

    (CuMatrix.fromDense(cpu_matrix), cpu_tau)
  }
}
