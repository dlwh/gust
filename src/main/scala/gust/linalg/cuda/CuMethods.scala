package gust.linalg.cuda

import breeze.linalg.support.{CanCollapseAxis, CanTranspose, CanSlice2}


import jcuda.jcublas.{cublasOperation, cublasHandle, JCublas2, cublasDiagType, cublasFillMode}
import gust.util.cuda
import jcuda.runtime.{cudaError, cudaMemcpyKind, cudaStream_t, JCuda}
import cuda._
import jcuda.jcurand.{curandRngType, curandGenerator}


/**
 * Created by piotrek on 21.05.2014.
 *
 * The algorithms are a Scala-port of the CUDA C version from OrangeOwlSolutions
 */
class CuMethods {

}

object CuMethods {
  // it will basically a collection of static methods.
  // perhaps they could be incorporated into CuMatrix?

  /** Gauss-Jordan solve of a linear system
    * as for now it's in-place and inefficient.
    *
    * The result ends up in the d_B vector */
  def solveSlow(d_A: CuMatrix[Double], d_B: CuMatrix[Double])(implicit handle: cublasHandle): Unit = {
    if (d_A.rows != d_B.rows) {
      println("Number of rows in A must be the same as number of rows in B")
      return
    }

    if (d_B.cols != 1) {
      println("B has to be a column vector")
      return
    }

    if (d_A.rows != d_A.cols) {
      println("A has to be a square matrix")
      return
    }

    //val d_A = A.copy
    //val d_B = B.copy
    val N = d_A.rows
    // val e_s = A.elemSize   // elems in A and B are the same size

    // this is a workaround for writing:   double alpha;   ..., &alpha   as in C
    val alpha = Array(0.0)
    val beta = Array(0.0)
    val A_ii = Array(0.0)
    val B_i = Array(0.0)
    val a_iiPtr = jcuda.Pointer.to(A_ii)
    val b_iPtr = jcuda.Pointer.to(B_i)
    var cuResult = 0  // just temporary, to check for errors

    val d_R = CuMatrix.zeros[Double](N, 1)  // a vector with ratios for every iteration

    for (i <- 0 until (N-1)) {

      // moving the current elem. on the diagonal to a variable, to use it later
      cuResult = JCuda.cudaMemcpy2D(a_iiPtr, d_A.majorStride * d_A.elemSize,
        d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(i, i) * d_A.elemSize),
        d_A.majorStride * d_A.elemSize, d_A.elemSize, 1, cudaMemcpyKind.cudaMemcpyDeviceToHost)


      if (Math.abs(A_ii(0)) < 1e-20) {
        println("A_ii: " + A_ii(0) + ", i: " + i)
        // this could be solved by pivoting...
        println("Division by (nearly) zero.")
        return
      }

      // copies the whole row "under" the element d_A(i, i) to d_R
      JCublas2.cublasDcopy(handle, N - (i+1), d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(i+1, i) * d_A.elemSize),
        1, d_R.data.toCuPointer.withByteOffset(d_R.linearIndex(i+1, 0) * d_R.elemSize), 1)


      alpha(0) = 1.0 / A_ii(0)
      // d_R *= alpha
      JCublas2.cublasDscal(handle, N - (i+1), jcuda.Pointer.to(alpha),
        d_R.data.toCuPointer.withByteOffset(d_R.linearIndex(i+1, 0) * d_R.elemSize), 1)

      JCuda.cudaMemcpy2D(b_iPtr, d_B.majorStride * d_B.elemSize,
        d_B.data.toCuPointer.withByteOffset(d_B.linearIndex(i, 0) * d_B.elemSize),
        d_B.majorStride * d_B.elemSize, d_B.elemSize, 1, cudaMemcpyKind.cudaMemcpyDeviceToHost)

      alpha(0) = -B_i(0)
      // d_B += d_R * alpha
      JCublas2.cublasDaxpy(handle, N - (i+1), jcuda.Pointer.to(alpha),
        d_R.data.toCuPointer.withByteOffset(d_R.linearIndex(i+1, 0) * d_R.elemSize), 1,
        d_B.data.toCuPointer.withByteOffset(d_B.linearIndex(i+1, 0) * d_B.elemSize), 1)

      alpha(0) = -1.0
      beta(0) = 1.0

      // update of the whole submatrix (col_vector * row_vector == matrix)
      JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
        N - (i+1), N - i, 1, jcuda.Pointer.to(alpha),
        d_R.data.toCuPointer.withByteOffset(d_R.linearIndex(i+1, 0) * d_R.elemSize), N,
        d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(i, i) * d_A.elemSize), N, jcuda.Pointer.to(beta),
        d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(i+1, i) * d_A.elemSize), N)

    }

    // solve the triangular system:
    JCublas2.cublasDtrsv(handle, cublasFillMode.CUBLAS_FILL_MODE_UPPER, cublasOperation.CUBLAS_OP_N,
      cublasDiagType.CUBLAS_DIAG_NON_UNIT, N,
      d_A.data.toCuPointer, d_A.majorSize,
      d_B.data.toCuPointer, 1)
  }


  /**
   * A faster version using stripe approach
   * @param d_A
   * @param d_B
   * @param handle
   * @return
   */
  def solve(d_A: CuMatrix[Double], d_B: CuMatrix[Double])(implicit handle: cublasHandle): Unit = {
    if (d_A.rows != d_B.rows) {
      println("Number of rows in A must be the same as number of rows in B")
      return
    }

    if (d_B.cols != 1) {
      println("B has to be a column vector")
      return
    }

    if (d_A.rows != d_A.cols) {
      println("A has to be a square matrix")
      return
    }

    val N = d_A.rows
    val matrixBlockSize = 2     // this is just for testing --> should be more
    val matrixStripeSize = 2
    val numBlocks = N / matrixBlockSize + (if (N % matrixBlockSize == 0) 0 else 1)

    val alpha = Array(0.0)
    val beta = Array(0.0)
    val A_ii = Array(0.0)
    val B_i = Array(0.0)
    val a_iiPtr = jcuda.Pointer.to(A_ii)
    val b_iPtr = jcuda.Pointer.to(B_i)

    val d_R = CuMatrix.zeros[Double](N, matrixBlockSize)


    for (iB <- 0 until numBlocks) {

      val i0 = iB * N / numBlocks
      val i1 = (iB+1) * N / numBlocks


      // iB.iB
      for (i <- i0 until i1) {
        JCuda.cudaMemcpy2D(a_iiPtr, d_A.majorStride * d_A.elemSize,
          d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(i, i) * d_A.elemSize),
          d_A.majorStride * d_A.elemSize, d_A.elemSize, 1, cudaMemcpyKind.cudaMemcpyDeviceToHost)

        if (Math.abs(A_ii(0)) < 1e-20) {
          println("A_ii: " + A_ii(0) + ", i: " + i)
          println("Division by (nearly) zero.")
          return
        }

        // cublasDcopy(handle, i1-(i+1), d_A+(i+1)+i*N, 1, d_R+(i+1)+(i-i0)*N, 1)
        JCublas2.cublasDcopy(handle, i1 - (i+1),
                             d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(i+1, i) * d_A.elemSize), 1,
                             d_R.data.toCuPointer.withByteOffset(d_R.linearIndex(i+1, i-i0) * d_R.elemSize), 1)

        beta(0) = 1.0 / A_ii(0)
        // cublasDscal(handle, i1-(i+1), &beta, d_R+(i+1)+(i-i0)*N, 1)
        JCublas2.cublasDscal(handle, i1 - (i+1), jcuda.Pointer.to(beta),
                             d_R.data.toCuPointer.withByteOffset(d_R.linearIndex(i+1, i-i0) * d_R.elemSize), 1)

        JCuda.cudaMemcpy2D(b_iPtr, d_B.majorStride * d_B.elemSize,
          d_B.data.toCuPointer.withByteOffset(d_B.linearIndex(i, 0) * d_B.elemSize),
          d_B.majorStride * d_B.elemSize, d_B.elemSize, 1, cudaMemcpyKind.cudaMemcpyDeviceToHost)

        beta(0) = -B_i(0)
        //cublasDaxpy(handle, i1-(i+1), &beta, d_R+(i+1)+(i-i0)*N, 1, d_B+(i+1), 1)
        JCublas2.cublasDaxpy(handle, i1 - (i + 1), jcuda.Pointer.to(beta),
                             d_R.data.toCuPointer.withByteOffset(d_R.linearIndex(i+1, i-i0) * d_R.elemSize), 1,
                             d_B.data.toCuPointer.withByteOffset(d_B.linearIndex(i+1, 0) * d_B.elemSize), 1)

        alpha(0) = -1.0
        beta(0) = 1.0
        // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, i1-(i+1), i1-i, 1, &alpha,
        // d_R+(i+1)+(i-i0)*N, N, d_A+i+i*N, N, &beta, d_A+(i+1)+i*N, N)
        JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, i1 - (i+1), i1 - i, 1,
                             jcuda.Pointer.to(alpha),
                             d_R.data.toCuPointer.withByteOffset(d_R.linearIndex(i+1, i-i0) * d_R.elemSize), N,
                             d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(i, i) * d_A.elemSize), N,
                             jcuda.Pointer.to(beta),
                             d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(i+1, i) * d_A.elemSize), N)
      }

      // iB.kB
      alpha(0) = -1.0
      beta(0) = 1.0

      for (kB <- (iB + 1) until numBlocks) {
        val k0 = kB * N / numBlocks
        val k1 = (kB + 1) * N / numBlocks

        val numStripes = (i1 - i0) / matrixStripeSize + (if ((i1 - i0) % matrixStripeSize == 0) 0 else 1)

        // for (int iQ=0; iQ<Num_Stripes; iQ++) {
        for (iQ <- 0 until numStripes) {
          val i_0 = i0 + iQ * (i1 - i0) / numStripes
          val i_1 = i0 + (iQ + 1) * (i1 - i0) / numStripes

          // for (int i=i_0; i<i_1; i++)
          for (i <- i_0 until i_1) {
            // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, i_1-(i+1), k1-k0, 1, &alpha,
            // d_R+(i+1)+(i-i0)*N, N, d_A+i+k0*N, N, &beta, d_A+(i+1)+k0*N, N)
            JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
              i_1 - (i + 1), k1 - k0, 1, jcuda.Pointer.to(alpha),
              d_R.data.toCuPointer.withByteOffset(d_R.linearIndex(i + 1, i - i0) * d_R.elemSize), N,
              d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(i, k0) * d_A.elemSize), N,
              jcuda.Pointer.to(beta),
              d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(i + 1, k0) * d_A.elemSize), N)
          }

          // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, i1-i_1, k1-k0, i_1-i_0, &alpha,
          // d_R+i_1+(i_0-i0)*N, N, d_A+i_0+k0*N, N, &beta, d_A+i_1+k0*N, N))
          JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
            i1 - i_1, k1 - k0, i_1 - i_0, jcuda.Pointer.to(alpha),
            d_R.data.toCuPointer.withByteOffset(d_R.linearIndex(i_1, i_0 - i0) * d_R.elemSize), N,
            d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(i_0, k0) * d_A.elemSize), N,
            jcuda.Pointer.to(beta),
            d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(i_1, k0) * d_A.elemSize), N)
        }
      }

      if (iB < numBlocks - 1) {
        // double* vectAii = (double*)malloc((i1-i0)*sizeof(double));
        // double* vectBi	= (double*)malloc((i1-i0)*sizeof(double));
        val vectAii = Array.ofDim[Double](i1 - i0)
        val vectBi = Array.ofDim[Double](i1 - i0)
        // cublasGetVector(i1-i0, sizeof(double), d_A+i0+i0*N, N+1, vectAii, 1)
        // puts the i1-i0 elems from diagonal in vectAii:
        JCublas2.cublasGetVector(i1-i0, d_A.elemSize.toInt,
                                 d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(i0, i0) * d_A.elemSize), N+1,
                                 jcuda.Pointer.to(vectAii), 1)
        // cublasGetVector(i1-i0, sizeof(double), d_B+i0, 1, vectBi, 1)
        JCublas2.cublasGetVector(i1-i0, d_B.elemSize.toInt,
                                 d_B.data.toCuPointer.withByteOffset(d_B.linearIndex(i0, 0) * d_B.elemSize), 1,
                                 jcuda.Pointer.to(vectBi), 1)


        // jB.iB
        for (jB <- (iB + 1) until numBlocks) {
          val j0 = jB * N / numBlocks
          val j1 = (jB + 1) * N / numBlocks

          val numStripes = (i1 - i0) / matrixStripeSize + (if ((i1 - i0) % matrixStripeSize == 0) 0 else 1)


          for (iQ <- 0 until numStripes) {
            val i_0 = i0 + iQ * (i1 - i0) / numStripes
            val i_1 = i0 + (iQ + 1) * (i1 - i0) / numStripes

            for (i <- i_0 until i_1) {

              // cublasDcopy(handle, j1-j0, d_A+j0+i*N, 1, d_R+j0+(i-i0)*N, 1)
              JCublas2.cublasDcopy(handle, j1-j0,
                                   d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(j0, i) * d_A.elemSize), 1,
                                   d_R.data.toCuPointer.withByteOffset(d_R.linearIndex(j0, i-i0) * d_R.elemSize), 1)

              beta(0) = 1.0 / vectAii(i-i0)
              // cublasDscal(handle, j1-j0, &beta, d_R+j0+(i-i0)*N , 1)
              JCublas2.cublasDscal(handle, j1-j0, jcuda.Pointer.to(beta),
                                   d_R.data.toCuPointer.withByteOffset(d_R.linearIndex(j0, i-i0) * d_R.elemSize), 1)

              beta(0) = -vectBi(i-i0)
              // cublasDaxpy(handle, j1-j0, &beta, d_R+j0+(i-i0)*N, 1, d_B+j0, 1)
              JCublas2.cublasDaxpy(handle, j1-j0, jcuda.Pointer.to(beta),
                                   d_R.data.toCuPointer.withByteOffset(d_R.linearIndex(j0, i-i0) * d_R.elemSize), 1,
                                   d_B.data.toCuPointer.withByteOffset(d_B.linearIndex(j0, 0) * d_B.elemSize), 1)

              alpha(0) = -1.0
              beta(0) = 1.0
              // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, j1-j0, i_1-i, 1, &alpha,
              // d_R+j0+(i-i0)*N, N, d_A+i+i*N, N, &beta, d_A+j0+i*N, N)
              JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
                                   j1-j0, i_1-i, 1, jcuda.Pointer.to(alpha),
                                   d_R.data.toCuPointer.withByteOffset(d_R.linearIndex(j0, i-i0) * d_R.elemSize), N,
                                   d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(i, i) * d_A.elemSize), N,
                                   jcuda.Pointer.to(beta),
                                   d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(j0, i) * d_A.elemSize), N)
            }

            alpha(0) = -1.0
            beta(0) = 1.0
            // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, j1-j0, i1-i_1, i_1-i_0, &alpha,
            // d_R+j0+(i_0-i0)*N, N, d_A+i_0+i_1*N, N, &beta, d_A+j0+i_1*N, N)
            JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
              j1-j0, i1-i_1, i_1-i_0, jcuda.Pointer.to(alpha),
              d_R.data.toCuPointer.withByteOffset(d_R.linearIndex(j0, i_0-i0) * d_R.elemSize), N,
              d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(i_0, i_1) * d_A.elemSize), N,
              jcuda.Pointer.to(beta),
              d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(j0, i_1) * d_A.elemSize), N)

          }
        }

        // jB.kB
        // for (int jB = iB + 1; jB < Num_Blocks; jB++) {
        for (jB <- (iB + 1) until numBlocks) {

          val j0 = jB * N / numBlocks
          val j1 = (jB + 1) * N / numBlocks

          for (kB <- (iB + 1) until numBlocks) {
            val k0 = kB * N / numBlocks
            val k1 = (kB + 1) * N / numBlocks

            alpha(0) = -1.0
            beta(0) = 1.0

            // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, j1-j0, k1-k0, i1-i0, &alpha,
            // d_R+j0, N, d_A+i0+k0*N, N, &beta, d_A+j0+k0*N, N)
            JCublas2.cublasDgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
                                 j1-j0, k1-k0, i1-i0, jcuda.Pointer.to(alpha),
                                 d_R.data.toCuPointer.withByteOffset(d_R.linearIndex(j0, 0) * d_R.elemSize), N,
                                 d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(i0, k0) * d_A.elemSize), N,
                                 jcuda.Pointer.to(beta),
                                 d_A.data.toCuPointer.withByteOffset(d_A.linearIndex(j0, k0) * d_A.elemSize), N)
          }

        }

      }
    }

    // solve the triangular system:
    JCublas2.cublasDtrsv(handle, cublasFillMode.CUBLAS_FILL_MODE_UPPER, cublasOperation.CUBLAS_OP_N,
      cublasDiagType.CUBLAS_DIAG_NON_UNIT, N,
      d_A.data.toCuPointer, d_A.majorSize,
      d_B.data.toCuPointer, 1)
  }
}