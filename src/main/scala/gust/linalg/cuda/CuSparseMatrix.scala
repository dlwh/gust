package gust.linalg.cuda

import breeze.linalg.DenseMatrix
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
                        val csrColInd: CuMatrix[Int])(implicit val sparseHandle: cusparseHandle) {


  def toCuMatrix(implicit blasHandle: cublasHandle, ct: ClassTag[V]): CuMatrix[V] = {
    val A = CuMatrix.create[V](rows, cols)
    val matType = ct.toString()
    val cuspFunc = if (matType.equals("Double")) JCusparse.cusparseDcsr2dense _
                   else if (matType.equals("Float")) JCusparse.cusparseScsr2dense _
                   else throw new UnsupportedOperationException("can't convert a matrix that's not Float nor Double")

    cuspFunc(sparseHandle, rows, cols, descr,
      csrVal.offsetPointer, csrRowPtr.offsetPointer, csrColInd.offsetPointer, A.offsetPointer, A.majorStride)

    A
  }

  def toDense(implicit blasHandle: cublasHandle, ct: ClassTag[V]): DenseMatrix[V] = this.toCuMatrix.toDense
}

object CuSparseMatrix {
  // TODO handle transpose here
  def fromCuMatrix[V](A: CuMatrix[V])(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle, ct: ClassTag[V]) = {
    // create matrix descriptor and some info:
    val descr = new cusparseMatDescr
    JCusparse.cusparseCreateMatDescr(descr)
    JCusparse.cusparseSetMatType(descr, cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL)
    JCusparse.cusparseSetMatIndexBase(descr, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO)
    JCusparse.cusparseSetMatDiagType(descr, cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT)

    // we have to decide which functions to use:
    val isOfTypeD = ct.toString().equals("Double")
    val isOfTypeS = ct.toString().equals("Float")

    val cuspFuncNnz = if (isOfTypeD) JCusparse.cusparseDnnz _
                      else if (isOfTypeS) JCusparse.cusparseSnnz _
                      else throw new UnsupportedOperationException("can't convert a matrix that's not Float nor Double")

    val cuspFuncDense2csr = if (isOfTypeD) JCusparse.cusparseDdense2csr _
                            else if (isOfTypeS) JCusparse.cusparseSdense2csr _
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

    new CuSparseMatrix[V](A.rows, A.cols, descr, csrValA, csrRowPtrA, csrColIndA)
  }

  def fromDense[V <: AnyVal](A: DenseMatrix[V])(implicit sparseHandle: cusparseHandle, blasHandle: cublasHandle, ct: ClassTag[V]) = {
    // create matrix descriptor and some info:
    val descr = new cusparseMatDescr
    JCusparse.cusparseCreateMatDescr(descr)
    JCusparse.cusparseSetMatType(descr, cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL)
    JCusparse.cusparseSetMatIndexBase(descr, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO)
    JCusparse.cusparseSetMatDiagType(descr, cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT)

    val d_A = CuMatrix.fromDense(A)

    // we have to decide which functions to use:
    val isOfTypeD = ct.toString().equals("Double")
    val isOfTypeS = ct.toString().equals("Float")

    val cuspFuncNnz = if (isOfTypeD) JCusparse.cusparseDnnz _
    else if (isOfTypeS) JCusparse.cusparseSnnz _
    else throw new UnsupportedOperationException("can't convert a matrix that's not Float nor Double")

    val cuspFuncDense2csr = if (isOfTypeD) JCusparse.cusparseDdense2csr _
    else if (isOfTypeS) JCusparse.cusparseSdense2csr _
    else throw new UnsupportedOperationException("can't convert a matrix that's not Float nor Double")

    // get the number of non-zero entries per row:
    val nnzPerRow = CuMatrix.create[Int](d_A.rows, 1)
    val nnzHost = Array(0)
    val nnzHostPtr= jcuda.Pointer.to(nnzHost)

    cuspFuncNnz(sparseHandle, cusparseDirection.CUSPARSE_DIRECTION_ROW,
      d_A.rows, d_A.cols, descr, d_A.offsetPointer, A.majorStride, nnzPerRow.offsetPointer, nnzHostPtr)


    // create the sparse matrix:
    val nnz = nnzHost(0)
    val csrValA = CuMatrix.create[V](nnz, 1)
    val csrRowPtrA = CuMatrix.create[Int](d_A.rows+1, 1)
    val csrColIndA = CuMatrix.create[Int](nnz, 1)

    cuspFuncDense2csr(sparseHandle, d_A.rows, d_A.cols, descr, d_A.offsetPointer, d_A.majorStride,
      nnzPerRow.offsetPointer, csrValA.offsetPointer, csrRowPtrA.offsetPointer, csrColIndA.offsetPointer)

    new CuSparseMatrix[V](d_A.rows, d_A.cols, descr, csrValA, csrRowPtrA, csrColIndA)
  }
}