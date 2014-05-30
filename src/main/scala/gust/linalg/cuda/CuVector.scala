package gust.linalg.cuda

import breeze.linalg.{norm, DenseVector, DenseMatrix, NumericOps}
import org.bridj.Pointer
import jcuda.jcublas.{cublasHandle, JCublas2}
import gust.util.cuda._
import breeze.generic.UFunc
import spire.algebra.VectorSpace
import breeze.math.{Semiring, MutableInnerProductSpace, MutableCoordinateSpace, CoordinateSpace}
import breeze.linalg.support.CanCreateZerosLike
import breeze.linalg.operators.{OpSet, OpMulScalar}
import scala.reflect.ClassTag
import jcuda.runtime.JCuda
import jcuda.jcurand.{curandRngType, curandGenerator}
import gust.util.cuda

/**
 * A gpu side dense vector
 *
 * @author dlwh
 **/
class CuVector[V](val data: Pointer[V],
                  val offset: Int,
                  val stride: Int,
                  val length: Int) extends NumericOps[CuVector[V]] {
  override def repr: CuVector[V] = this
  def this(length: Int)(implicit ct: ClassTag[V]) = this(cuda.allocate[V](length), 0, 1, length)

  def elemSize = data.getIO.getTargetSize
  def offsetPointer = data.toCuPointer.withByteOffset(elemSize * offset)

  def size = length

  def toDense = {
    val arrayData = Pointer.allocateArray(data.getIO, size)


    JCublas2.cublasGetVector(length, elemSize.toInt, data.toCuPointer.withByteOffset(elemSize * offset), stride, arrayData.toCuPointer, 1)

    new DenseVector(arrayData.toArray)
  }

  def toMatrix = {
    new CuMatrix[V](length, 1, data, offset, stride, false)
  }


}

object CuVector {


  /**
   * The standard way to create an empty matrix, size is length
   */
  def zeros[V](length: Int)(implicit ct: ClassTag[V]): CuVector[V] = {
    val mat = new CuVector[V](length)

    JCuda.cudaMemset(mat.data.toCuPointer, 0, mat.size * mat.elemSize)

    mat
  }

  /**
   * The standard way to create an empty matrix, size is length
   */
  def ones[V](length: Int)(implicit ct: ClassTag[V], semiring: Semiring[V], canSet: OpSet.InPlaceImpl2[CuVector[V], V]): CuVector[V] = {
    val mat = new CuVector[V](length)

    mat := semiring.one


    mat
  }

  /**
   * Doesn't zero the matrix.
   */
  def create[V](length: Int)(implicit ct: ClassTag[V]): CuVector[V] = {
    val mat = new CuVector[V](length)
    JCuda.cudaMemset(mat.data.toCuPointer, 0, mat.size * mat.elemSize)

    mat
  }

  def rand(length: Int)(implicit blas: cublasHandle) = {
    import jcuda.jcurand.JCurand._
    val mat = new CuVector[Float](length)
    val generator = new curandGenerator()
    curandCreateGenerator(generator, curandRngType.CURAND_RNG_PSEUDO_DEFAULT)
    curandSetPseudoRandomGeneratorSeed(generator, 1234)

    curandGenerateUniform(generator, mat.data.toCuPointer, length)
    curandDestroyGenerator(generator)

    mat
  }

  implicit def canCreateZerosLike[V:ClassTag]:CanCreateZerosLike[CuVector[V], CuVector[V]] = new CanCreateZerosLike[CuVector[V], CuVector[V]] {
    override def apply(from: CuVector[V]): CuVector[V] = {
      zeros(from.length)
    }
  }


//  implicit val vspaceFloat = MutableInnerProductSpace.make[CuVector[Float], Float]

}
