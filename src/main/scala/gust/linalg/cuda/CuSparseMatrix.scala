package gust.linalg.cuda

import breeze.linalg.operators.{OpSolveMatrixBy, OpMulMatrix, OpSub, OpAdd}
import breeze.linalg.support.CanTranspose
import breeze.linalg.{cholesky, LU, NumericOps, DenseMatrix}
import com.nativelibs4java.opencl.{CLQueue, CLContext}
import gust.util.cuda
import jcuda.jcublas.{JCublas2, cublasHandle}
import jcuda.jcusparse._
import jcuda.runtime.{cudaMemcpyKind, JCuda}
import spire.syntax.cfor
import scala.reflect.ClassTag

/**
 * Created by Piotr on 2014-07-18.
 *
 * NOTE: Matrix is stored in CSC format (following Breeze), but the cusparse API
 * favours the CSR format. This is why the cusparse functions see the matrix as CSR,
 * which is effectively a transpose. This may cause some confusion, for example:
 * multiplication is done in reverse order (because B' * A' = (AB)'). However, the methods
 * aim to be transparent to the end-user so it's not that bad after all.
 */
class CuSparseMatrix[V](val rows: Int, val cols: Int,
                        val descr: cusparseMatDescr,
                        val cscVal: CuMatrix[V],
                        val cscColPtr: CuMatrix[Int],
                        val cscRowInd: CuMatrix[Int],
                        val isTranspose: Boolean = false)(implicit val sparseHandle: cusparseHandle) extends NumericOps[CuSparseMatrix[V]] {

  def size = rows * cols

  def elemSize = cscVal.data.getIO.getTargetSize

  def nnz = cscVal.size

  def toCuMatrix(implicit blasHandle: cublasHandle, ct: ClassTag[V]): CuMatrix[V] = {
    val A = CuMatrix.create[V](rows, cols)
    val cuspFunc = if (A.elemSize == 8) JCusparse2.cusparseDcsc2dense _ // if (matType.equals("Double")) JCusparse2.cusparseDcsr2dense _
                   else if (A.elemSize == 4) JCusparse2.cusparseScsc2dense _ // else if (matType.equals("Float")) JCusparse2.cusparseScsr2dense _
                   else throw new UnsupportedOperationException("can't convert a matrix that's not Float nor Double")

    cuspFunc(sparseHandle, rows, cols, descr,
      cscVal.offsetPointer, cscRowInd.offsetPointer, cscColPtr.offsetPointer, A.offsetPointer, A.majorStride)

    if (isTranspose) A.t else A
  }

  def toDense(implicit blasHandle: cublasHandle, ct: ClassTag[V]): DenseMatrix[V] = this.toCuMatrix.toDense

  def copy(implicit blasHandle: cublasHandle, ct: ClassTag[V]) = {
    val descrC = new cusparseMatDescr
    JCusparse2.cusparseCreateMatDescr(descrC)
    JCusparse2.cusparseSetMatType(descrC, cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL)
    JCusparse2.cusparseSetMatIndexBase(descrC, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO)
    JCusparse2.cusparseSetMatDiagType(descrC, cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT)

    val cscValC = CuMatrix.create[V](cscVal.rows, 1); cscValC.writeFrom(cscVal)
    val cscColPtrC = CuMatrix.create[Int](cscColPtr.rows, 1); cscColPtrC.writeFrom(cscColPtr)
    val cscRowIndC = CuMatrix.create[Int](cscRowInd.rows, 1); cscRowIndC.writeFrom(cscRowInd)

    new CuSparseMatrix[V](rows, cols, descrC, cscValC, cscColPtrC,
      cscRowIndC, isTranspose)
  }

  // perform the transpose explicitly:
  def transpose(implicit blasHandle: cublasHandle, ct: ClassTag[V]) = {
    val descrT = new cusparseMatDescr
    JCusparse2.cusparseCreateMatDescr(descrT)
    JCusparse2.cusparseSetMatType(descrT, cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL)
    JCusparse2.cusparseSetMatIndexBase(descrT, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO)
    JCusparse2.cusparseSetMatDiagType(descrT, cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT)

    val csrVal = CuMatrix.create[V](nnz, 1)
    val csrColInd = CuMatrix.create[Int](nnz, 1)
    val csrRowPtr = CuMatrix.create[Int](rows+1, 1)

    val cuspOp = if (elemSize == 8) JCusparse2.cusparseDcsr2csc _
                 else if (elemSize == 4) JCusparse2.cusparseScsr2csc _
                 else throw new UnsupportedOperationException("Can't transpose a matrix with elem sizes different than 4 and 8")

    cuspOp(sparseHandle, cols, rows, nnz, cscVal.offsetPointer, cscColPtr.offsetPointer, cscRowInd.offsetPointer,
      csrVal.offsetPointer, csrColInd.offsetPointer, csrRowPtr.offsetPointer,
      cusparseAction.CUSPARSE_ACTION_NUMERIC, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO)

    new CuSparseMatrix[V](cols, rows, descrT, csrVal, csrRowPtr,
      csrColInd, isTranspose)
  }

  def release() {
    JCusparse2.cusparseDestroyMatDescr(descr)
    cscVal.release()
    cscColPtr.release()
    cscRowInd.release()
  }

  override def repr = this

  override def toString = "cscVal:  " + this.cscVal.t +  "cscRowInd:  " + this.cscRowInd.t + "cscColPtr:  " + this.cscColPtr.t

}

object CuSparseMatrix extends CuSparseOps {
  this: CuSparseMatrix.type =>

  def fromCuMatrix[V](A: CuMatrix[V])(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle, ct: ClassTag[V]) = {
    // create matrix descriptor and some info:
    val descr = new cusparseMatDescr
    JCusparse2.cusparseCreateMatDescr(descr)
    JCusparse2.cusparseSetMatType(descr, cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL)
    JCusparse2.cusparseSetMatIndexBase(descr, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO)
    JCusparse2.cusparseSetMatDiagType(descr, cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT)

    // we have to decide which functions to use:
    val isOfTypeD = A.elemSize == 8 //ct.toString().equals("Double")
    val isOfTypeS = A.elemSize == 4 //ct.toString().equals("Float")

    // here we're using JCusparse because JCusparse2.cusparse<t>nnz appears not to be working
    val cuspFuncNnz = if (isOfTypeD) JCusparse.cusparseDnnz _
                      else if (isOfTypeS) JCusparse.cusparseSnnz _
                      else throw new UnsupportedOperationException("can't convert a matrix that's not Float nor Double")

    val cuspFuncDense2csc = if (isOfTypeD) JCusparse2.cusparseDdense2csc _
                            else if (isOfTypeS) JCusparse2.cusparseSdense2csc _
                            else throw new UnsupportedOperationException("can't convert a matrix that's not Float nor Double")

    // get the number of non-zero entries per row:
    val nnzPerCol = CuMatrix.create[Int](A.cols, 1)
    val nnzHost = Array(0)
    val nnzHostPtr= jcuda.Pointer.to(nnzHost)

    cuspFuncNnz(sparseHandle, cusparseDirection.CUSPARSE_DIRECTION_COLUMN,
         A.rows, A.cols, descr, A.offsetPointer, A.majorStride, nnzPerCol.offsetPointer, nnzHostPtr)


    // create the sparse matrix:
    val nnz = nnzHost(0)
    val cscValA = CuMatrix.create[V](nnz, 1)
    val cscColPtrA = CuMatrix.create[Int](A.cols+1, 1)
    val cscRowIndA = CuMatrix.create[Int](nnz, 1)

    cuspFuncDense2csc(sparseHandle, A.rows, A.cols, descr, A.offsetPointer, A.majorStride,
      nnzPerCol.offsetPointer, cscValA.offsetPointer, cscRowIndA.offsetPointer, cscColPtrA.offsetPointer)

    new CuSparseMatrix[V](A.rows, A.cols, descr, cscValA, cscColPtrA, cscRowIndA, A.isTranspose)
  }

  def fromDense[V <: AnyVal](A: DenseMatrix[V])(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle, ct: ClassTag[V]) = {
    CuSparseMatrix.fromCuMatrix(CuMatrix.fromDense(A))
  }


  implicit def canTranspose[V](implicit blasHandle: cublasHandle, ct: ClassTag[V]): CanTranspose[CuSparseMatrix[V], CuSparseMatrix[V]] = {
    new CanTranspose[CuSparseMatrix[V], CuSparseMatrix[V]] {
      def apply(from: CuSparseMatrix[V]) = {
        from.transpose
      }
    }
  }
}

trait CuSparseOps {
  implicit def CuSpMatrixFAddCuSpMatrixF(implicit blas: cublasHandle): OpAdd.Impl2[CuSparseMatrix[Float], CuSparseMatrix[Float], CuSparseMatrix[Float]] =
    new OpAdd.Impl2[CuSparseMatrix[Float], CuSparseMatrix[Float], CuSparseMatrix[Float]] {
    def apply(a: CuSparseMatrix[Float], b: CuSparseMatrix[Float]): CuSparseMatrix[Float] = {
      import a.sparseHandle

      CuWrapperMethods.sparseSgeam(1.0f, a, 1.0f, b)
    }
  }

  implicit def CuSpMatrixDAddCuSpMatrixD(implicit blas: cublasHandle): OpAdd.Impl2[CuSparseMatrix[Double], CuSparseMatrix[Double], CuSparseMatrix[Double]] =
    new OpAdd.Impl2[CuSparseMatrix[Double], CuSparseMatrix[Double], CuSparseMatrix[Double]] {
    def apply(a: CuSparseMatrix[Double], b: CuSparseMatrix[Double]): CuSparseMatrix[Double] = {
      import a.sparseHandle

      CuWrapperMethods.sparseDgeam(1.0, a, 1.0, b)
    }
  }

  implicit def CuSpMatrixFSubCuSpMatrixF(implicit blas: cublasHandle): OpSub.Impl2[CuSparseMatrix[Float], CuSparseMatrix[Float], CuSparseMatrix[Float]] =
    new OpSub.Impl2[CuSparseMatrix[Float], CuSparseMatrix[Float], CuSparseMatrix[Float]] {
    def apply(a: CuSparseMatrix[Float], b: CuSparseMatrix[Float]): CuSparseMatrix[Float] = {
      import a.sparseHandle

      CuWrapperMethods.sparseSgeam(1.0f, a, -1.0f, b)
    }
  }

  implicit def CuSpMatrixDSubCuSpMatrixD(implicit blas: cublasHandle): OpSub.Impl2[CuSparseMatrix[Double], CuSparseMatrix[Double], CuSparseMatrix[Double]] =
    new OpSub.Impl2[CuSparseMatrix[Double], CuSparseMatrix[Double], CuSparseMatrix[Double]] {
    def apply(a: CuSparseMatrix[Double], b: CuSparseMatrix[Double]): CuSparseMatrix[Double] = {
      import a.sparseHandle

      CuWrapperMethods.sparseDgeam(1.0, a, -1.0, b)
    }
  }

  implicit def CuSpMatrixFMulCuSpMatrixF(implicit blas: cublasHandle): OpMulMatrix.Impl2[CuSparseMatrix[Float], CuSparseMatrix[Float], CuSparseMatrix[Float]] =
    new OpMulMatrix.Impl2[CuSparseMatrix[Float], CuSparseMatrix[Float], CuSparseMatrix[Float]] {
    def apply(a: CuSparseMatrix[Float], b: CuSparseMatrix[Float]): CuSparseMatrix[Float] = {
      import a.sparseHandle

      CuWrapperMethods.sparseSgemm(a, b)
    }
  }

  implicit def CuSpMatrixDMulCuSpMatrixD(implicit blas: cublasHandle): OpMulMatrix.Impl2[CuSparseMatrix[Double], CuSparseMatrix[Double], CuSparseMatrix[Double]] =
    new OpMulMatrix.Impl2[CuSparseMatrix[Double], CuSparseMatrix[Double], CuSparseMatrix[Double]] {
    def apply(a: CuSparseMatrix[Double], b: CuSparseMatrix[Double]): CuSparseMatrix[Double] = {
      import a.sparseHandle

      CuWrapperMethods.sparseDgemm(a, b)
    }
  }

  implicit def CuSpMatrixFMulCuVectorF(implicit blas: cublasHandle): OpMulMatrix.Impl2[CuSparseMatrix[Float], CuMatrix[Float], CuMatrix[Float]] =
    new OpMulMatrix.Impl2[CuSparseMatrix[Float], CuMatrix[Float], CuMatrix[Float]] {
      def apply(a: CuSparseMatrix[Float], b: CuMatrix[Float]): CuMatrix[Float] = {
        import a.sparseHandle

        CuWrapperMethods.sparseSgemv(a, b)
      }
    }

  implicit def CuSpMatrixDMulCuVectorD(implicit blas: cublasHandle): OpMulMatrix.Impl2[CuSparseMatrix[Double], CuMatrix[Double], CuMatrix[Double]] =
    new OpMulMatrix.Impl2[CuSparseMatrix[Double], CuMatrix[Double], CuMatrix[Double]] {
      def apply(a: CuSparseMatrix[Double], b: CuMatrix[Double]): CuMatrix[Double] = {
        import a.sparseHandle

        CuWrapperMethods.sparseDgemv(a, b)
      }
    }

  implicit def CuSpMatrixFMulScalarF(implicit blas: cublasHandle): OpMulMatrix.Impl2[CuSparseMatrix[Float], Float, CuSparseMatrix[Float]] =
    new OpMulMatrix.Impl2[CuSparseMatrix[Float], Float, CuSparseMatrix[Float]] {
      def apply(a: CuSparseMatrix[Float], s: Float): CuSparseMatrix[Float] = {
        JCublas2.cublasSscal(blas, a.nnz, jcuda.Pointer.to(Array(s)), a.cscVal.offsetPointer, 1)
        a
      }
    }

  implicit def CuSpMatrixDMulScalarD(implicit blas: cublasHandle): OpMulMatrix.Impl2[CuSparseMatrix[Double], Double, CuSparseMatrix[Double]] =
    new OpMulMatrix.Impl2[CuSparseMatrix[Double], Double, CuSparseMatrix[Double]] {
      def apply(a: CuSparseMatrix[Double], s: Double): CuSparseMatrix[Double] = {
        JCublas2.cublasDscal(blas, a.nnz, jcuda.Pointer.to(Array(s)), a.cscVal.offsetPointer, 1)
        a
      }
    }


  implicit def canSolveCuSpMatrixFloat(implicit blas: cublasHandle): OpSolveMatrixBy.Impl2[CuSparseMatrix[Float], CuMatrix[Float], CuMatrix[Float]] =
    new OpSolveMatrixBy.Impl2[CuSparseMatrix[Float], CuMatrix[Float], CuMatrix[Float]] {
      def apply(a: CuSparseMatrix[Float], b: CuMatrix[Float]) = {
        import a.sparseHandle

        CuSolve.sparseSolveFloat(a, b)
      }
    }

  implicit def canSolveCuSpMatrixDouble(implicit blas: cublasHandle): OpSolveMatrixBy.Impl2[CuSparseMatrix[Double], CuMatrix[Double], CuMatrix[Double]] =
    new OpSolveMatrixBy.Impl2[CuSparseMatrix[Double], CuMatrix[Double], CuMatrix[Double]] {
      def apply(a: CuSparseMatrix[Double], b: CuMatrix[Double]) = {
        import a.sparseHandle

        CuSolve.sparseSolveDouble(a, b)
      }
    }

  /**
   * Incomplete LU factorization without pivoting. I'm not really sure whether this implicit should even be here.
   * It's not a "real" LU and it may give the false impression.
   */
  implicit def canLUFloat(implicit blasHandle: cublasHandle): LU.Impl[CuSparseMatrix[Float], (CuSparseMatrix[Float], CuSparseMatrix[Float])] =
    new LU.Impl[CuSparseMatrix[Float], (CuSparseMatrix[Float], CuSparseMatrix[Float])] {
      def apply(_a: CuSparseMatrix[Float])= {
        import _a.sparseHandle

        val d_LU = CuLU.incompleteLUFloat(_a)
        CuLU.incompleteLUFactorsFloat(d_LU)
      }
    }

  implicit def canLUDouble(implicit blasHandle: cublasHandle): LU.Impl[CuSparseMatrix[Double], (CuSparseMatrix[Double], CuSparseMatrix[Double])] =
    new LU.Impl[CuSparseMatrix[Double], (CuSparseMatrix[Double], CuSparseMatrix[Double])] {
      def apply(_a: CuSparseMatrix[Double])= {
        import _a.sparseHandle

        val d_LU = CuLU.incompleteLUDouble(_a)
        CuLU.incompleteLUFactorsDouble(d_LU)
      }
    }

  implicit def canCholeskyFloat(implicit blasHandle: cublasHandle): cholesky.Impl[CuSparseMatrix[Float], CuSparseMatrix[Float]] =
    new cholesky.Impl[CuSparseMatrix[Float], CuSparseMatrix[Float]] {
      def apply(_a: CuSparseMatrix[Float])= {
        import _a.sparseHandle

        CuCholesky.incompleteCholFactorFloat(CuCholesky.incompleteCholeskyFloat(_a))
      }
    }

  implicit def canCholeskyDouble(implicit blasHandle: cublasHandle): cholesky.Impl[CuSparseMatrix[Double], CuSparseMatrix[Double]] =
    new cholesky.Impl[CuSparseMatrix[Double], CuSparseMatrix[Double]] {
      def apply(_a: CuSparseMatrix[Double])= {
        import _a.sparseHandle

        CuCholesky.incompleteCholFactorDouble(CuCholesky.incompleteCholeskyDouble(_a))
      }
    }
}