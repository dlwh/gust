package snap.linalg.cuda


import breeze.linalg.operators.OpSet
import breeze.linalg._
import breeze.linalg.support.CanTranspose
import breeze.linalg.support.CanSlice2
import breeze.util.ArrayUtil
import breeze.storage.DefaultArrayValue
import org.bridj.{PointerIO, Pointer}
import scala.reflect.ClassTag

import jcuda.jcublas.{cublasOperation, cublasHandle, JCublas2}
import snap.util.{CanRepresentAs, cuda}
import jcuda.runtime.{cudaMemcpyKind, cudaStream_t, JCuda}
import jcuda.driver.CUstream
import cuda._

/**
 * TODO
 *
 * @author dlwh
 **/
class CuMatrix[V](val rows: Int,
                  val cols: Int,
                  val data: Pointer[V],
                  val offset: Int,
                  val majorStride: Int,
                  val isTranspose: Boolean = false)(implicit val blas: cublasHandle) extends NumericOps[CuMatrix[V]] {
  /** Creates a matrix with the specified data array, rows, and columns. Data must be column major */
  def this(rows: Int, cols: Int, data: Pointer[V], offset: Int = 0)(implicit blas: cublasHandle) = this(rows, cols, data, offset, rows)
  /** Creates a matrix with the specified data array, rows, and columns. */
  def this(rows: Int, cols: Int)(implicit blas: cublasHandle, ct: ClassTag[V]) = this(rows, cols, cuda.allocate[V](rows * cols))

  def size = rows * cols

  /** Calculates the index into the data array for row and column */
  final def linearIndex(row: Int, col: Int): Int = {
    if(isTranspose)
      offset + col + row * majorStride
    else
      offset + row + col * majorStride
  }

  def repr = this

  /*
  override def equals(p1: Any) = p1 match {
    case x: CuMatrix[_] =>

      // todo: make this faster in obvious cases
      rows == x.rows && cols == x.cols && (valuesIterator sameElements x.valuesIterator )

    case _ => false
  }
  */

  def majorSize = if(isTranspose) rows else cols

  def activeSize = size

  def footprint = majorSize * majorStride

  def isActive(i: Int) = true
  def allVisitableIndicesActive = true

  def elemSize = data.getIO.getTargetSize

  def writeFromDense(b: DenseMatrix[V]): Int = {
    require(b.rows == this.rows, "Matrices must have same number of rows")
    require(b.cols == this.cols, "Matrices must have same number of columns")

    if(isTranspose) {
      return this.t.writeFromDense(b.t)
    }

    val _b = if(b.isTranspose) b.copy else b

    val bPtr = cuda.cuPointerToArray(_b.data)

    val (width, height) = if(isTranspose) (cols, rows) else (rows, cols)

    assert(majorStride >= width, majorStride + " " + width)
    assert(_b.majorStride >= width)

    JCuda.cudaMemcpy2D(data.toCuPointer.withByteOffset(offset * elemSize),
      majorStride * elemSize,
      bPtr.withByteOffset(offset * elemSize),
      _b.majorStride * elemSize,
      width * elemSize,
      height,
      cudaMemcpyKind.cudaMemcpyHostToDevice
    )

    JCuda.cudaFreeHost(bPtr)

  }

  private def isGapless = (!this.isTranspose && this.majorStride == this.rows) || (this.isTranspose && this.majorStride == this.cols)


  def writeFrom(b: CuMatrix[V])(implicit stream: CUstream = new CUstream()) = {
    require(b.rows == this.rows, "Matrices must have same number of rows")
    require(b.cols == this.cols, "Matrices must have same number of columns")

    val aPtr = data.toCuPointer.withByteOffset(offset * elemSize)
    val bPtr = b.data.toCuPointer.withByteOffset(offset * elemSize)

    val (width, height) = if(isTranspose) (cols, rows) else (rows, cols)

    if(b.isGapless && this.isGapless && b.isTranspose == this.isTranspose)  {
      JCuda.cudaMemcpyAsync(aPtr, bPtr, size * elemSize, cudaMemcpyKind.cudaMemcpyDeviceToDevice, new cudaStream_t(stream))
    } else if(b.isTranspose == this.isTranspose) {
      JCuda.cudaMemcpy2DAsync(aPtr,
        majorStride * elemSize,
        bPtr,
        b.majorStride * elemSize,
        width * elemSize,
        height,
        cudaMemcpyKind.cudaMemcpyDeviceToDevice,
        new cudaStream_t(stream)
      )

    } else {
      val op = if(elemSize == 4) {
        JCublas2.cublasSgeam _
      } else if(elemSize == 8) {
        JCublas2.cublasDgeam _
      } else {
        throw new UnsupportedOperationException("can't do a copy-transpose with elems that are not of size 4 or 8")
      }

      blas.withStream(stream) {
        op(blas, cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_T,
          width, height,
          CuMatrix.hostOne,
          bPtr,
          b.majorStride,
          CuMatrix.hostZero,
          bPtr, b.majorStride, aPtr, majorStride)
      }


    }



  }

  /*
  def assignAsync(b: V)(implicit stream: CUstream = new CUstream(), cast: CanRepresentAs[V, Int]) = {
    require(elemSize == 4)
    val (width, height) = if(isTranspose) (cols, rows) else (rows, cols)
    JCuda.cudaMemset2DAsync(data.toCuPointer, majorStride, cast.convert(b), width, height, stream)
  }
  */

  /** Forcibly releases the buffer. Note that other slices will be invalidated! */
  def release() = {
    data.release()
  }

  def toDense = {
    val arrayData = Pointer.allocateArray(data.getIO, size)

    val (_r, _c) = if(isTranspose) (cols, rows) else (rows, cols)

   JCublas2.cublasGetMatrix(_r, _c, elemSize.toInt, data.toCuPointer, majorStride, arrayData.toCuPointer, _r)

    new DenseMatrix(rows, cols, arrayData.getArray.asInstanceOf[Array[V]], 0, _r, isTranspose)
  }

  def copy: Matrix[V] = ???
}

object CuMatrix extends LowPriorityNativeMatrix {
  /**
   * The standard way to create an empty matrix, size is rows * cols
   */
  def zeros[V](rows: Int, cols: Int)(implicit ct: ClassTag[V], dav: DefaultArrayValue[V], blas: cublasHandle): CuMatrix[V] = {
    val mat = new CuMatrix[V](rows, cols)

    JCuda.cudaMemset(mat.data.toCuPointer, 0, mat.size * mat.elemSize)

    mat
  }


  /*


  // slices
  implicit def canSliceRow[V:ClassTag]: CanSlice2[CuMatrix[V], Int, ::.type, CuMatrix[V]] = {
    new CanSlice2[CuMatrix[V], Int, ::.type, CuMatrix[V]] {
      def apply(m: CuMatrix[V], row: Int, ignored: ::.type) = {
        import m.queue
        if(row < 0 || row >= m.rows) throw new ArrayIndexOutOfBoundsException("Row must be in bounds for slice!")
        if(!m.isTranspose)
          new CuMatrix(1, m.cols, m.data, m.offset + row, m.majorStride)
        else
          new CuMatrix(1, m.cols, m.data, m.offset + row * m.cols, 1)
      }
    }
  }

  implicit def canSliceCol[V:ClassTag]: CanSlice2[CuMatrix[V], ::.type, Int, CuMatrix[V]] = {
    new CanSlice2[CuMatrix[V], ::.type, Int, CuMatrix[V]] {
      def apply(m: CuMatrix[V], ignored: ::.type, col: Int) = {
        import m.queue
        if(col < 0 || col >= m.cols) throw new ArrayIndexOutOfBoundsException("Column must be in bounds for slice!")
        if(!m.isTranspose)
          new CuMatrix(m.rows, 1, m.data, col * m.majorStride + m.offset)
        else
          new CuMatrix(1, m.cols, m.data, offset = m.offset + col, majorStride = m.majorStride)
      }
    }
  }

  implicit def canSliceRows[V:ClassTag]: CanSlice2[CuMatrix[V], Range, ::.type, CuMatrix[V]] = {
    new CanSlice2[CuMatrix[V], Range, ::.type, CuMatrix[V]] {
      def apply(m: CuMatrix[V], rows: Range, ignored: ::.type) = {
        import m.queue
        if(rows.isEmpty) new CuMatrix(0, 0, m.data, 0, 0)
        else if(!m.isTranspose) {
          assert(rows.head >= 0)
          assert(rows.last < m.rows, s"last row ${rows.last} is bigger than rows ${m.rows}")
          require(rows.step == 1, "Sorry, we can't support row ranges with step sizes other than 1")
          val first = rows.head
          new CuMatrix(rows.length, m.cols, m.data, m.offset + first, m.majorStride)
        } else {
          assert(rows.head >= 0)
          assert(rows.last < m.rows)
          canSliceCols.apply (m.t, ::, rows).t
        }
      }
    }
  }

  implicit def canSliceCols[V:ClassTag]: CanSlice2[CuMatrix[V], ::.type, Range, CuMatrix[V]] = {
    new CanSlice2[CuMatrix[V], ::.type, Range, CuMatrix[V]] {
      def apply(m: CuMatrix[V], ignored: ::.type, cols: Range) = {
        import m.queue
        if(cols.isEmpty) new CuMatrix(m.rows, 0, m.data, 0, 1)
        else if(!m.isTranspose) {
          assert(cols.head >= 0)
          assert(cols.last < m.cols, cols.last + " " + m.cols)
          val first = cols.head
          new CuMatrix(m.rows, cols.length, m.data, m.offset + first * m.majorStride, m.majorStride * cols.step)
        } else {
          canSliceRows.apply(m.t, cols, ::).t
        }
      }
    }
  }

  implicit def canSliceColsAndRows[V:ClassTag]: CanSlice2[CuMatrix[V], Range, Range, CuMatrix[V]] = {
    new CanSlice2[CuMatrix[V], Range, Range, CuMatrix[V]] {
      def apply(m: CuMatrix[V], rows: Range, cols: Range) = {
        import m.queue
        if(rows.isEmpty || cols.isEmpty) new CuMatrix(0, 0, m.data, 0, 1)
        else if(!m.isTranspose) {
          assert(cols.head >= 0)
          assert(cols.last < m.cols)
          assert(rows.head >= 0)
          assert(rows.last < m.rows)
          require(rows.step == 1, "Sorry, we can't support row ranges with step sizes other than 1 for non transposed matrices")
          val first = cols.head
          new CuMatrix(rows.length, cols.length, m.data, m.offset + first * m.rows + rows.head, m.majorStride * cols.step)(m.queue, implicitly)
        } else {
          require(cols.step == 1, "Sorry, we can't support col ranges with step sizes other than 1 for transposed matrices")
          canSliceColsAndRows.apply(m.t, cols, rows).t
        }
      }
    }
  }



  implicit def canSlicePartOfRow[V:ClassTag]: CanSlice2[CuMatrix[V], Int, Range, CuMatrix[V]] = {
    new CanSlice2[CuMatrix[V], Int, Range, CuMatrix[V]] {
      def apply(m: CuMatrix[V], row: Int, cols: Range) = {
        import m.queue
        if(row < 0  || row > m.rows) throw new IndexOutOfBoundsException("Slice with out of bounds row! " + row)
        if(cols.isEmpty) new CuMatrix(0, 0, m.data, 0, 1)
        else if(!m.isTranspose) {
          val first = cols.head
          new CuMatrix(1, cols.length, m.data, m.offset + first * m.rows + row, m.majorStride * cols.step)
        } else {
          require(cols.step == 1, "Sorry, we can't support col ranges with step sizes other than 1 for transposed matrices")
          canSlicePartOfCol.apply(m.t, cols, row).t
        }
      }
    }
  }

  implicit def canSlicePartOfCol[V:ClassTag]: CanSlice2[CuMatrix[V], Range, Int, CuMatrix[V]] = {
    new CanSlice2[CuMatrix[V], Range, Int, CuMatrix[V]] {
      def apply(m: CuMatrix[V], rows: Range, col: Int) = {
        import m.queue
        if(rows.isEmpty) new CuMatrix(0, 0, m.data, 0)
        else if(!m.isTranspose) {
          new CuMatrix(col * m.rows + m.offset + rows.head, 1, m.data, rows.step, rows.length)
        } else {
          val m2 = canSlicePartOfRow.apply(m.t, col, rows).t
          m2(::, 0)
        }
      }
    }
  }

  /*
  implicit def canMapValues[V, R:ClassTag] = {
    new CanMapValues[CuMatrix[V],V,R,CuMatrix[R]] {
      override def map(from : CuMatrix[V], fn : (V=>R)) = {
        val data = new Array[R](from.size)
        var j = 0
        var off = 0
        while (j < from.cols) {
          var i = 0
          while(i < from.rows) {
            data(off) = fn(from(i, j))
            off += 1
            i += 1
          }
          j += 1
        }
        new CuMatrix[R](from.rows, from.cols, data)
      }

      override def mapActive(from : CuMatrix[V], fn : (V=>R)) =
        map(from, fn)
    }
  }


  implicit def canTransformValues[V]:CanTransformValues[CuMatrix[V], V, V] = {
    new CanTransformValues[CuMatrix[V], V, V] {
      def transform(from: CuMatrix[V], fn: (V) => V) {
        var j = 0
        while (j < from.cols) {
          var i = 0
          while(i < from.rows) {
            from(i, j) = fn(from(i, j))
            i += 1
          }
          j += 1
        }
      }

      def transformActive(from: CuMatrix[V], fn: (V) => V) {
        transform(from, fn)
      }
    }
  }

  implicit def canMapKeyValuePairs[V, R:ClassTag] = {
    new CanMapKeyValuePairs[CuMatrix[V],(Int,Int),V,R,CuMatrix[R]] {
      override def map(from : CuMatrix[V], fn : (((Int,Int),V)=>R)) = {
        val data = new Array[R](from.data.length)
        var j = 0
        var off = 0
        while (j < from.cols) {
          var i = 0
          while(i < from.rows) {
            data(off) = fn(i -> j, from(i, j))
            off += 1
            i += 1
          }
          j += 1
        }
        new CuMatrix(from.rows, from.cols, data)
      }

      override def mapActive(from : CuMatrix[V], fn : (((Int,Int),V)=>R)) =
        map(from, fn)
    }
  }
  */
  */

  implicit def canTranspose[V]: CanTranspose[CuMatrix[V], CuMatrix[V]] = {
    new CanTranspose[CuMatrix[V], CuMatrix[V]] {
      def apply(from: CuMatrix[V]) = {
        new CuMatrix(data = from.data, offset = from.offset, cols = from.rows, rows = from.cols, majorStride = from.majorStride, isTranspose = !from.isTranspose)(from.blas)
      }
    }
  }

  /*
  implicit def canTransposeComplex: CanTranspose[CuMatrix[Complex], CuMatrix[Complex]] = {
    new CanTranspose[CuMatrix[Complex], CuMatrix[Complex]] {
      def apply(from: CuMatrix[Complex]) = {
        new CuMatrix(data = from.data map { _.conjugate },
          offset = from.offset,
          cols = from.rows,
          rows = from.cols,
          majorStride = from.majorStride,
          isTranspose = !from.isTranspose)
      }
    }
  }
  */


  /**
   * Maps the columns into a new dense matrix
   * @tparam V
   * @tparam R
   * @return
  implicit def canMapRows[V:ClassTag:DefaultArrayValue]: CanCollapseAxis[CuMatrix[V], Axis._0.type, CuMatrix[V], CuMatrix[V], CuMatrix[V]]  = new CanCollapseAxis[CuMatrix[V], Axis._0.type, CuMatrix[V], CuMatrix[V], CuMatrix[V]] {
    def apply(from: CuMatrix[V], axis: Axis._0.type)(f: (CuMatrix[V]) => CuMatrix[V]): CuMatrix[V] = {
      var result:CuMatrix[V] = null
      for(c <- 0 until from.cols) {
        val col = f(from(::, c))
        if(result eq null) {
          result = CuMatrix.zeros[V](col.length, from.cols)
        }
        result(::, c) := col
      }
      if(result eq null){
        CuMatrix.zeros[V](0, from.cols)
      } else {
        result
      }
    }
  }

  /**
   * Returns a numRows CuMatrix
   * @tparam V
   * @tparam R
   * @return
   */
  implicit def canMapCols[V:ClassTag:DefaultArrayValue] = new CanCollapseAxis[CuMatrix[V], Axis._1.type, CuMatrix[V], CuMatrix[V], CuMatrix[V]] {
    def apply(from: CuMatrix[V], axis: Axis._1.type)(f: (CuMatrix[V]) => CuMatrix[V]): CuMatrix[V] = {
      var result:CuMatrix[V] = null
      val t = from.t
      for(r <- 0 until from.rows) {
        val row = f(t(::, r))
        if(result eq null) {
          result = CuMatrix.zeros[V](from.rows, row.length)
        }
        result.t apply (::, r) := row
      }
      result
    }
  }


  //  implicit val setMM_D: BinaryUpdateOp[CuMatrix[Double], CuMatrix[Double], OpSet] = new SetCuMCuMOp[Double]
  //  implicit val setMM_F: BinaryUpdateOp[CuMatrix[Float], CuMatrix[Float], OpSet]  = new SetCuMCuMOp[Float]
  //  implicit val setMM_I: BinaryUpdateOp[CuMatrix[Int], CuMatrix[Int], OpSet]  = new SetCuMCuMOp[Int]

/*
  implicit def canGaxpy[V: Semiring]: CanAxpy[V, CuMatrix[V], CuMatrix[V]] = {
    new CanAxpy[V, CuMatrix[V], CuMatrix[V]] {
      val ring = implicitly[Semiring[V]]
      def apply(s: V, b: CuMatrix[V], a: CuMatrix[V]) {
        require(a.rows == b.rows, "Vector row dimensions must match!")
        require(a.cols == b.cols, "Vector col dimensions must match!")

        var i = 0
        while (i < a.rows) {
          var j = 0
          while (j < a.cols) {
            a(i, j) = ring.+(a(i, j), ring.*(s, b(i, j)))
            j += 1
          }
          i += 1
        }
      }
    }
  }
   */
   */


  protected val hostOnePtr = Pointer.pointerToFloat(1)

  protected val hostOne = hostOnePtr.toCuPointer


  protected val hostZeroPtr = Pointer.pointerToFloat(0)

  protected val hostZero = hostZeroPtr.toCuPointer
}

trait LowPriorityNativeMatrix1 {
  //  class SetMMOp[@specialized(Int, Double, Float) V] extends BinaryUpdateOp[CuMatrix[V], Matrix[V], OpSet] {
  //    def apply(a: CuMatrix[V], b: Matrix[V]) {
  //      require(a.rows == b.rows, "Matrixs must have same number of rows")
  //      require(a.cols == b.cols, "Matrixs must have same number of columns")
  //
  //      // slow path when we don't have a trivial matrix
  //      val ad = a.data
  //      var c = 0
  //      while(c < a.cols) {
  //        var r = 0
  //        while(r < a.rows) {
  //          ad(a.linearIndex(r, c)) = b(r, c)
  //          r += 1
  //        }
  //        c += 1
  //      }
  //    }
  //  }



  //  class SetDMVOp[@specialized(Int, Double, Float) V] extends BinaryUpdateOp[CuMatrix[V], Vector[V], OpSet] {
  //    def apply(a: CuMatrix[V], b: Vector[V]) {
  //      require(a.rows == b.length && a.cols == 1 || a.cols == b.length && a.rows == 1, "CuMatrix must have same number of rows, or same number of columns, as CuMatrix, and the other dim must be 1.")
  //      val ad = a.data
  //      var i = 0
  //      var c = 0
  //      while(c < a.cols) {
  //        var r = 0
  //        while(r < a.rows) {
  //          ad(a.linearIndex(r, c)) = b(i)
  //          r += 1
  //          i += 1
  //        }
  //        c += 1
  //      }
  //    }
  //  }
  //
  //  implicit def setMM[V]: BinaryUpdateOp[CuMatrix[V], Matrix[V], OpSet] = new SetMMOp[V]
  //  implicit def setMV[V]: BinaryUpdateOp[CuMatrix[V], Vector[V], OpSet] = new SetDMVOp[V]
}

trait LowPriorityNativeMatrix extends LowPriorityNativeMatrix1 {

  class SetCuMCuMVOp[V] extends OpSet.InPlaceImpl2[CuMatrix[V], CuMatrix[V]] {
    def apply(a: CuMatrix[V], b: CuMatrix[V]) {
      a.writeFrom(b.asInstanceOf[CuMatrix[V]])
    }
  }

  implicit def SetCuMDMOp[V <: AnyVal]: OpSet.InPlaceImpl2[CuMatrix[V], DenseMatrix[V]] = new  OpSet.InPlaceImpl2[CuMatrix[V], DenseMatrix[V]] {
    def apply(a: CuMatrix[V], b: DenseMatrix[V]) {
      a.writeFromDense(b)
    }
  }



  implicit object setCuMCuMFloat extends SetCuMCuMVOp[Float]
  implicit object setCuMCuMLong extends SetCuMCuMVOp[Long]
  implicit object setCuMCuMInt extends SetCuMCuMVOp[Int]
  implicit object setCuMCuMDouble extends SetCuMCuMVOp[Double]

  /*
  class SetDMDVOp[@specialized(Int, Double, Float) V] extends BinaryUpdateOp[CuMatrix[V], CuMatrix[V], OpSet] {
    def apply(a: CuMatrix[V], b: CuMatrix[V]) {
      require(a.rows == b.length && a.cols == 1 || a.cols == b.length && a.rows == 1, "CuMatrix must have same number of rows, or same number of columns, as CuMatrix, and the other dim must be 1.")
      val ad = a.data
      val bd = b.data
      var c = 0
      var boff = b.offset
      while(c < a.cols) {
        var r = 0
        while(r < a.rows) {
          ad(a.linearIndex(r, c)) = bd(boff)
          r += 1
          boff += b.stride
        }
        c += 1
      }
    }
  }


  implicit object SetMSFloatOp extends OpSet.InPlaceImpl2[CuMatrix[Float], Float] {
    def apply(a: CuMatrix[Float], b: Float) {
      val zmk = ZeroMemoryKernel()(a.queue.getContext)
      import a.queue
      // nicely shaped matrix
      if( (!a.isTranspose && a.majorStride == a.rows)  ||(a.isTranspose && a.majorStride == a.cols)) {
        val ev = zmk.fillMemory(a.data, b, a.offset, a.rows * a.cols)
        ev.waitFor()
      } else {
        zmk.shapedFill(a, b).waitFor()
      }
    }
  }

  implicit object SetMSIntOp extends OpSet.InPlaceImpl2[CuMatrix[Int], Int] {
    def apply(a: CuMatrix[Int], b: Int) {
      val zmk = ZeroMemoryKernel()(a.queue.getContext)
      import a.queue
      // nicely shaped matrix
      if( (!a.isTranspose && a.majorStride == a.rows)  ||(a.isTranspose && a.majorStride == a.cols)) {
        val ev = zmk.fillMemory(a.data.asCLFloatBuffer(), java.lang.Float.intBitsToFloat(b), a.offset, a.rows * a.cols)
        ev.waitFor()
      } else {
        zmk.shapedFill(a.asInstanceOf[CuMatrix[Float]], java.lang.Float.intBitsToFloat(b)).waitFor()
      }
    }
  }

  */

  private def transposeOp(a: DenseMatrix[Double]): Int = {
    if (a.isTranspose) cublasOperation.CUBLAS_OP_T else cublasOperation.CUBLAS_OP_N
  }


}





