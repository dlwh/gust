package gust.linalg.cuda


import breeze.linalg.operators._
import breeze.linalg._
import breeze.linalg.support.{CanCollapseAxis, CanTranspose, CanSlice2}
import org.bridj.Pointer
import scala.reflect.ClassTag

import jcuda.jcublas.{cublasOperation, cublasHandle, JCublas2}
import gust.util.cuda
import jcuda.runtime.{cudaError, cudaMemcpyKind, cudaStream_t, JCuda}
import jcuda.driver.CUstream
import cuda._
import jcuda.jcurand.{curandRngType, curandGenerator}
import breeze.math.{Semiring, Ring}
import breeze.numerics._
import breeze.generic.UFunc

/**
 * Created by piotrek on 21.05.2014.
 */
class CuMethods {

}

object CuMethods {
  // it will basically a collection of static methods.
  // perhaps they could be incorporated into CuMatrix?

  /** Gauss-Jordan solve of a linear system
    * as for now it's in-place and inefficient.
    * And only forward elimination phase... */
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

    // TODO: make this more "functional style"
    for (i <- 0 to (N-2)) {

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
  }

}