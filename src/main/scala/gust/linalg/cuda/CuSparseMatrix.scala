package gust.linalg.cuda

import breeze.linalg.{NumericOps, DenseMatrix}
import gust.util.cuda
import jcuda.jcublas.cublasHandle
import jcuda.jcusparse._
import scala.reflect.ClassTag

/**
 * Created by Piotr on 2014-07-18.
 */
class CuSparseMatrix[V](val rows: Int, val cols: Int,
                        val descr: cusparseMatDescr,
                        val csrVal: CuMatrix[V],
                        val csrRowPtr: CuMatrix[Int],
                        val csrColInd: CuMatrix[Int],
                        val isTranspose: Boolean = false)(implicit val sparseHandle: cusparseHandle) extends NumericOps[CuSparseMatrix[V]] {

  def size = rows * cols

  def elemSize = csrVal.data.getIO.getTargetSize

  def toCuMatrix(implicit blasHandle: cublasHandle, ct: ClassTag[V]): CuMatrix[V] = {
    val A = CuMatrix.create[V](rows, cols)
    val matType = ct.toString()
    val cuspFunc = if (matType.equals("Double")) JCusparse2.cusparseDcsr2dense _
                   else if (matType.equals("Float")) JCusparse2.cusparseScsr2dense _
                   else throw new UnsupportedOperationException("can't convert a matrix that's not Float nor Double")

    cuspFunc(sparseHandle, rows, cols, descr,
      csrVal.offsetPointer, csrRowPtr.offsetPointer, csrColInd.offsetPointer, A.offsetPointer, A.majorStride)

    if (isTranspose) A.t else A
  }

  def toDense(implicit blasHandle: cublasHandle, ct: ClassTag[V]): DenseMatrix[V] = this.toCuMatrix.toDense

  def copy(implicit blasHandle: cublasHandle, ct: ClassTag[V]) = {
    val descrC = new cusparseMatDescr
    JCusparse2.cusparseCreateMatDescr(descrC)
    JCusparse2.cusparseSetMatType(descrC, cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL)
    JCusparse2.cusparseSetMatIndexBase(descrC, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO)
    JCusparse2.cusparseSetMatDiagType(descrC, cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT)

    val csrValC = CuMatrix.create[V](csrVal.rows, 1); csrValC.writeFrom(csrVal)
    val csrRowPtrC = CuMatrix.create[Int](csrRowPtr.rows, 1); csrRowPtrC.writeFrom(csrRowPtr)
    val csrColIndC = CuMatrix.create[Int](csrColInd.rows, 1); csrColIndC.writeFrom(csrColInd)

    new CuSparseMatrix[V](rows, cols, descrC, csrValC, csrRowPtrC,
      csrColIndC, isTranspose)
  }

  def release() {
    JCusparse2.cusparseDestroyMatDescr(descr)
    csrVal.release()
    csrRowPtr.release()
    csrColInd.release()
  }

  override def repr = this

  override def toString = "csrVal:  " + this.csrVal.t +  "csrColInd:  " + this.csrColInd.t + "csrRowPtr:  " + this.csrRowPtr.t

}

object CuSparseMatrix {

  def fromCuMatrix[V](A: CuMatrix[V])(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle, ct: ClassTag[V]) = {
    // create matrix descriptor and some info:
    val descr = new cusparseMatDescr
    JCusparse2.cusparseCreateMatDescr(descr)
    JCusparse2.cusparseSetMatType(descr, cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL)
    JCusparse2.cusparseSetMatIndexBase(descr, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO)
    JCusparse2.cusparseSetMatDiagType(descr, cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT)

    // we have to decide which functions to use:
    val isOfTypeD = ct.toString().equals("Double")
    val isOfTypeS = ct.toString().equals("Float")

    // here we're using JCusparse because JCusparse2.cusparse<t>nnz appears not to be working
    val cuspFuncNnz = if (isOfTypeD) JCusparse.cusparseDnnz _
                      else if (isOfTypeS) JCusparse.cusparseSnnz _
                      else throw new UnsupportedOperationException("can't convert a matrix that's not Float nor Double")

    val cuspFuncDense2csr = if (isOfTypeD) JCusparse2.cusparseDdense2csr _
                            else if (isOfTypeS) JCusparse2.cusparseSdense2csr _
                            else throw new UnsupportedOperationException("can't convert a matrix that's not Float nor Double")

    // get the number of non-zero entries per row:
    val nnzPerRow = CuMatrix.create[Int](A.rows, 1)
    val nnzHost = Array(0)
    val nnzHostPtr= jcuda.Pointer.to(nnzHost)

    cuspFuncNnz(sparseHandle, cusparseDirection.CUSPARSE_DIRECTION_ROW,
         A.rows, A.cols, descr, A.offsetPointer, A.majorStride, nnzPerRow.offsetPointer, nnzHostPtr)


    // create the sparse matrix:
    val nnz = nnzHost(0)
    val csrValA = CuMatrix.create[V](nnz, 1)
    val csrRowPtrA = CuMatrix.create[Int](A.rows+1, 1)
    val csrColIndA = CuMatrix.create[Int](nnz, 1)

    cuspFuncDense2csr(sparseHandle, A.rows, A.cols, descr, A.offsetPointer, A.majorStride,
      nnzPerRow.offsetPointer, csrValA.offsetPointer, csrRowPtrA.offsetPointer, csrColIndA.offsetPointer)

    new CuSparseMatrix[V](A.rows, A.cols, descr, csrValA, csrRowPtrA, csrColIndA, A.isTranspose)
  }

  def fromDense[V <: AnyVal](A: DenseMatrix[V])(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle, ct: ClassTag[V]) = {
    // create matrix descriptor and some info:
    val descr = new cusparseMatDescr
    JCusparse2.cusparseCreateMatDescr(descr)
    JCusparse2.cusparseSetMatType(descr, cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL)
    JCusparse2.cusparseSetMatIndexBase(descr, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO)
    JCusparse2.cusparseSetMatDiagType(descr, cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT)

    val d_A = CuMatrix.fromDense(A)

    // we have to decide which functions to use:
    val isOfTypeD = ct.toString().equals("Double")
    val isOfTypeS = ct.toString().equals("Float")

    // same as above with cusparse<t>nnz
    val cuspFuncNnz = if (isOfTypeD) JCusparse.cusparseDnnz _
    else if (isOfTypeS) JCusparse.cusparseSnnz _
    else throw new UnsupportedOperationException("can't convert a matrix that's not Float nor Double")

    val cuspFuncDense2csr = if (isOfTypeD) JCusparse2.cusparseDdense2csr _
    else if (isOfTypeS) JCusparse2.cusparseSdense2csr _
    else throw new UnsupportedOperationException("can't convert a matrix that's not Float nor Double")

    // get the number of non-zero entries per row:
    val nnzPerRow = CuMatrix.create[Int](d_A.rows, 1)
    val nnzHost = Array(0)
    val nnzHostPtr = jcuda.Pointer.to(nnzHost)

    cuspFuncNnz(sparseHandle, cusparseDirection.CUSPARSE_DIRECTION_ROW,
      d_A.rows, d_A.cols, descr, d_A.offsetPointer, A.majorStride, nnzPerRow.offsetPointer, nnzHostPtr)


    // create the sparse matrix:
    val nnz = nnzHost(0)
    val csrValA = CuMatrix.create[V](nnz, 1)
    val csrRowPtrA = CuMatrix.create[Int](d_A.rows+1, 1)
    val csrColIndA = CuMatrix.create[Int](nnz, 1)

    cuspFuncDense2csr(sparseHandle, d_A.rows, d_A.cols, descr, d_A.offsetPointer, d_A.majorStride,
      nnzPerRow.offsetPointer, csrValA.offsetPointer, csrRowPtrA.offsetPointer, csrColIndA.offsetPointer)

    new CuSparseMatrix[V](d_A.rows, d_A.cols, descr, csrValA, csrRowPtrA, csrColIndA, d_A.isTranspose)
  }
}

trait CuSparseOps {

}