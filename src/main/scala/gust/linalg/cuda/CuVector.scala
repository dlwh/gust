package gust.linalg.cuda

import breeze.linalg._
import org.bridj.Pointer
import jcuda.jcublas.{cublasOperation, cublasHandle, JCublas2}
import gust.util.cuda._
import breeze.generic.UFunc
import spire.algebra.VectorSpace
import breeze.math.{Semiring, MutableInnerProductVectorSpace}
import breeze.linalg.support.{CanSlice, CanSlice2, CanCopy, CanCreateZerosLike}
import breeze.linalg.operators._
import scala.reflect.ClassTag
import jcuda.runtime.{cudaStream_t, cudaMemcpyKind, JCuda}
import jcuda.jcurand.{curandRngType, curandGenerator}
import gust.util.cuda
import breeze.numerics._
import breeze.generic.UFunc.{UImpl, UImpl2, InPlaceImpl2}
import jcuda.driver.CUstream
import breeze.stats.distributions.{RandBasis, Rand}

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



  def writeFrom(b: CuVector[V])(implicit stream: CUstream = new CUstream()) = {
    require(b.length == this.length, "Matrices must have same number of length")

    val aPtr = offsetPointer
    val bPtr = b.offsetPointer

    val (width, height) = (1, b.length)

    if(b.stride == 1 && this.stride == 1) {
      JCuda.cudaMemcpyAsync(aPtr, bPtr, size * elemSize, cudaMemcpyKind.cudaMemcpyDeviceToDevice, new cudaStream_t(stream))
    } else {
      JCuda.cudaMemcpy2DAsync(aPtr,
        stride * elemSize,
        bPtr,
        b.stride * elemSize,
        width * elemSize,
        height,
        cudaMemcpyKind.cudaMemcpyDeviceToDevice,
        new cudaStream_t(stream)
      )

    }

  }

  def writeFromDense(b: DenseVector[V]): Int = {
    require(b.length == this.length, "Matrices must have same number of length")


    val bPtr = cuda.cuPointerToArray(b.data)


    JCuda.cudaMemcpy2D(data.toCuPointer.withByteOffset(offset * elemSize),
      stride * elemSize,
      bPtr.withByteOffset(offset * elemSize),
      b.stride * elemSize,
      1 * elemSize,
      length,
      cudaMemcpyKind.cudaMemcpyHostToDevice
    )

    JCuda.cudaFreeHost(bPtr)

  }


  /**
   * Method for slicing that is tuned for Matrices.
   * @return
   */
  def apply[Slice1, Result](slice1: Slice1)(implicit canSlice: CanSlice[CuVector[V], Slice1, Result]) = {
    canSlice(this, slice1)
  }

  def release() = {
    data.release()
  }


}

object CuVector extends CuVectorFuns with CuVectorLowPrio {



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

  def fromDense[V<:AnyVal](mat: DenseVector[V])(implicit ct: ClassTag[V], blas: cublasHandle) = {
    val g = new CuVector[V](mat.length)
    g := mat
    g
  }

  /**
   * Doesn't zero the matrix.
   */
  def create[V](length: Int)(implicit ct: ClassTag[V]): CuVector[V] = {
    val mat = new CuVector[V](length)
    JCuda.cudaMemset(mat.data.toCuPointer, 0, mat.size * mat.elemSize)

    mat
  }

  def rand(length: Int)(implicit rand: RandBasis = Rand) = {
    import jcuda.jcurand.JCurand._
    val mat = new CuVector[Float](length)
    val generator = new curandGenerator()
    curandCreateGenerator(generator, curandRngType.CURAND_RNG_PSEUDO_DEFAULT)
    curandSetPseudoRandomGeneratorSeed(generator, rand.randInt.draw())

    curandGenerateUniform(generator, mat.data.toCuPointer, length)
    curandDestroyGenerator(generator)

    mat
  }

  implicit def canCreateZerosLike[V:ClassTag]:CanCreateZerosLike[CuVector[V], CuVector[V]] = new CanCreateZerosLike[CuVector[V], CuVector[V]] {
    override def apply(from: CuVector[V]): CuVector[V] = {
      zeros(from.length)
    }
  }

  implicit def canCopy[V:ClassTag](implicit handle: cublasHandle):CanCopy[CuVector[V]] = new CanCopy[CuVector[V]] {
    override def apply(from: CuVector[V]): CuVector[V] = {
      val a = create(from.length)
      a := from
      a
    }
  }

  implicit def canDotFloat(implicit handle: cublasHandle):OpMulInner.Impl2[CuVector[Float], CuVector[Float], Float] = {
    new OpMulInner.Impl2[CuVector[Float], CuVector[Float], Float] {
      override def apply(v: CuVector[Float], v2: CuVector[Float]): Float = {
        require(v.length == v2.length, "Length mismatch!")
        val ptr = cuda.allocateHost[Float](1)
        JCublas2.cublasSdot(handle, v.length, v.offsetPointer, v.stride, v2.offsetPointer, v2.stride, ptr.toCuPointer)
        val res = ptr.getFloat
        ptr.release()
        res
      }
    }

  }

  implicit def canAxpyFloat(implicit handle: cublasHandle):scaleAdd.InPlaceImpl3[CuVector[Float], Float, CuVector[Float]] = {
    new scaleAdd.InPlaceImpl3[CuVector[Float], Float, CuVector[Float]] {
      override def apply(v: CuVector[Float], v2: Float, v3: CuVector[Float]): Unit = {
        require(v.length == v3.length, "Length mismatch!")
        val ptr = cuda.allocateHost[Float](1)
        ptr.setFloat(v2)
        JCublas2.cublasSaxpy(handle, v.length, ptr.toCuPointer, v3.offsetPointer, v3.stride, v.offsetPointer, v.stride)
      }
    }

  }


  implicit def normImplFloat(implicit handle: cublasHandle): norm.Impl2[CuVector[Float], Double, Double] = new norm.Impl2[CuVector[Float], Double, Double] {
    override def apply(v: CuVector[Float], v2: Double): Double = {
      if(v2 == 2.0) {
        math.sqrt(v dot v)
      } else {
        ???
      }
    }
  }

  implicit def canDiagFloat(implicit handle: cublasHandle): diag.Impl[CuVector[Float], CuMatrix[Float]] =
    new diag.Impl[CuVector[Float], CuMatrix[Float]] {
      def apply(v: CuVector[Float]) = {
        val cm = CuMatrix.zeros[Float](v.size, v.size)
        JCublas2.cublasScopy(handle, v.size, v.offsetPointer, v.stride, cm.offsetPointer, cm.majorStride + 1)

        cm
      }
    }

  implicit def canDiagDouble(implicit handle: cublasHandle): diag.Impl[CuVector[Double], CuMatrix[Double]] =
    new diag.Impl[CuVector[Double], CuMatrix[Double]] {
      def apply(v: CuVector[Double]) = {
        val cm = CuMatrix.zeros[Double](v.size, v.size)
        JCublas2.cublasDcopy(handle, v.size, v.offsetPointer, v.stride, cm.offsetPointer, cm.majorStride + 1)

        cm
      }
    }

  implicit def vspaceFloat(implicit handle: cublasHandle): MutableInnerProductVectorSpace[CuVector[Float], Float] = {
    MutableInnerProductVectorSpace.make[CuVector[Float], Float]
  }

  // slicing
  implicit def canSlice[V]: CanSlice[CuVector[V], Range, CuVector[V]] = __canSlice.asInstanceOf[CanSlice[CuVector[V], Range, CuVector[V]]]

  private val __canSlice: CanSlice[CuVector[Any], Range, CuVector[Any]]  = {
    new CanSlice[CuVector[Any], Range, CuVector[Any]] {
      def apply(v: CuVector[Any], re: Range): CuVector[Any] = {

        val r = re.getRangeWithoutNegativeIndexes( v.length )

        require(r.isEmpty || r.last < v.length)
        require(r.isEmpty || r.start >= 0)
        new CuVector(v.data, offset = v.offset + v.stride * r.start, stride = v.stride * r.step, length = r.length)
      }
    }
  }






}

trait CuVectorFuns extends CuVectorKernels { this: CuVector.type =>
  implicit val kernelsFloat: KernelBroker[Float] = new KernelBroker[Float]("float")

  implicit def acosImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[acos.type]("acos")
  implicit def asinImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[asin.type]("asin")
  implicit def atanImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[atan.type]("atan")

  implicit def acoshImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[acosh.type]("acosh")
  implicit def asinhImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[asinh.type]("asinh")
  implicit def atanhImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[atanh.type]("atanh")

  implicit def cosImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[cos.type]("cos")
  implicit def sinImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[sin.type]("sin")
  implicit def tanImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[tan.type]("tan")

  implicit def coshImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[cosh.type]("cosh")
  implicit def sinhImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[sinh.type]("sinh")
  implicit def tanhImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[tanh.type]("tanh")

  implicit def cbrtImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[cbrt.type]("cbrt")
  implicit def ceilImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[ceil.type]("ceil")
  //  implicit def cospiImpl[T](implicit broker: CuMapKernels[T]) =  broker.implFor[cospi.type]("cospi")
  implicit def erfcImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[erfc.type]("erfc")
  implicit def erfcinvImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[erfcinv.type]("erfcinv")
  implicit def erfImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[erf.type]("erf")
  implicit def erfinvImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[erfinv.type]("erfinv")
  implicit def expImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[exp.type]("exp")
  implicit def expm1Impl[T](implicit broker: KernelBroker[T]) =  broker.implFor[expm1.type]("expm1")
  implicit def fabsImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[abs.type]("fabs")
  implicit def floorImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[floor.type]("floor")
  implicit def j0Impl[T](implicit broker: KernelBroker[T]) =  broker.implFor[Bessel.i0.type]("j0")
  implicit def j1Impl[T](implicit broker: KernelBroker[T]) =  broker.implFor[Bessel.i1.type]("j1")
  implicit def lgammaImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[lgamma.type]("lgamma")
  implicit def log10Impl[T](implicit broker: KernelBroker[T]) =  broker.implFor[log10.type]("log10")
  implicit def log1pImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[log1p.type]("log1p")
  //  implicit def log2Impl[T](implicit broker: CuMapKernels[T]) =  broker.implFor[log2.type]("log2")
  //  implicit def logbImpl[T](implicit broker: CuMapKernels[T]) =  broker.implFor[logb.type]("logb")
  implicit def logImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[log.type]("log")
  implicit def sqrtImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[sqrt.type]("sqrt")
  implicit def rintImpl[T](implicit broker: KernelBroker[T]) =  broker.implFor[rint.type]("rint")
  //  implicit def truncImpl[T](implicit broker: CuMapKernels[T]) =  broker.implFor[trunc.type]("trunc")

  implicit def acosIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[acos.type]("acos")
  implicit def asinIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[asin.type]("asin")
  implicit def atanIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[atan.type]("atan")

  implicit def acoshIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[acosh.type]("acosh")
  implicit def asinhIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[asinh.type]("asinh")
  implicit def atanhIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[atanh.type]("atanh")

  implicit def cosIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[cos.type]("cos")
  implicit def sinIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[sin.type]("sin")
  implicit def tanIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[tan.type]("tan")

  implicit def coshIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[cosh.type]("cosh")
  implicit def sinhIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[sinh.type]("sinh")
  implicit def tanhIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[tanh.type]("tanh")

  implicit def cbrtIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[cbrt.type]("cbrt")
  implicit def ceilIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[ceil.type]("ceil")
  //  implicit def cospiIntoImpl[T](implicit broker: CuMapKernels[T]) =  broker.inPlaceImplFor[cospi.type]("cospi")
  implicit def erfcIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[erfc.type]("erfc")
  implicit def erfcinvIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[erfcinv.type]("erfcinv")
  implicit def erfIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[erf.type]("erf")
  implicit def erfinvIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[erfinv.type]("erfinv")
  implicit def expIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[exp.type]("exp")
  implicit def expm1IntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[expm1.type]("expm1")
  implicit def fabsIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[abs.type]("fabs")
  implicit def floorIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[floor.type]("floor")
  implicit def j0IntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[Bessel.i0.type]("j0")
  implicit def j1IntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[Bessel.i1.type]("j1")
  implicit def lgammaIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[lgamma.type]("lgamma")
  implicit def log10IntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[log10.type]("log10")
  implicit def log1pIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[log1p.type]("log1p")
  //  implicit def log2IntoImpl[T](implicit broker: CuMapKernels[T]) =  broker.inPlaceImplFor[log2.type]("log2")
  //  implicit def logbIntoImpl[T](implicit broker: CuMapKernels[T]) =  broker.inPlaceImplFor[logb.type]("logb")
  implicit def logIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[log.type]("log")
  implicit def sqrtIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[sqrt.type]("sqrt")
  implicit def rintIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImplFor[rint.type]("rint")

  implicit def negateImpl[T](implicit broker: KernelBroker[T]): UImpl[OpNeg.type, CuVector[T], CuVector[T]] =  broker.implFor[OpNeg.type]("negate")

  implicit def addImpl[T](implicit broker: KernelBroker[T]): UImpl2[OpAdd.type, CuVector[T], CuVector[T], CuVector[T]] =  broker.impl2For[OpAdd.type]("add")
  implicit def subImpl[T](implicit broker: KernelBroker[T]): UImpl2[OpSub.type, CuVector[T], CuVector[T], CuVector[T]] =  broker.impl2For[OpSub.type]("sub")
  implicit def mulImpl[T](implicit broker: KernelBroker[T]): UImpl2[OpMulScalar.type, CuVector[T], CuVector[T], CuVector[T]] =  broker.impl2For[OpMulScalar.type]("mul")
  implicit def divImpl[T](implicit broker: KernelBroker[T]) =  broker.impl2For[OpDiv.type]("div")
  implicit def modImpl[T](implicit broker: KernelBroker[T]) =  broker.impl2For[OpMod.type]("mod")
  implicit def maxImpl[T](implicit broker: KernelBroker[T]) =  broker.impl2For[max.type]("max")
  implicit def minImpl[T](implicit broker: KernelBroker[T]) =  broker.impl2For[min.type]("min")
  implicit def powImpl[T](implicit broker: KernelBroker[T]) =  broker.impl2For[OpPow.type]("pow")

  implicit def addIntoImpl[T](implicit broker: KernelBroker[T]): InPlaceImpl2[OpAdd.type, CuVector[T], CuVector[T]] =  broker.inPlaceImpl2For[OpAdd.type]("add")
  implicit def subIntoImpl[T](implicit broker: KernelBroker[T]): InPlaceImpl2[OpSub.type, CuVector[T], CuVector[T]] =  broker.inPlaceImpl2For[OpSub.type]("sub")
  implicit def mulIntoImpl[T](implicit broker: KernelBroker[T]): InPlaceImpl2[OpMulScalar.type, CuVector[T], CuVector[T]] =  broker.inPlaceImpl2For[OpMulScalar.type]("mul")
  implicit def divIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImpl2For[OpDiv.type]("div")
  implicit def modIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImpl2For[OpMod.type]("mod")
  implicit def maxIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImpl2For[max.type]("max")
  implicit def minIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImpl2For[min.type]("min")
  implicit def powIntoImpl[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImpl2For[OpPow.type]("pow")

  implicit def addIntoImpl_S[T](implicit broker: KernelBroker[T]): InPlaceImpl2[OpAdd.type, CuVector[T], T] =  broker.inPlaceImpl2For_v_s[OpAdd.type]("add")
  implicit def subIntoImpl_S[T](implicit broker: KernelBroker[T]): InPlaceImpl2[OpSub.type, CuVector[T], T] =  broker.inPlaceImpl2For_v_s[OpSub.type]("sub")
  implicit def mulIntoImpl_S[T](implicit broker: KernelBroker[T]): InPlaceImpl2[OpMulScalar.type, CuVector[T], T] =  broker.inPlaceImpl2For_v_s[OpMulScalar.type]("mul")
  implicit def divIntoImpl_S[T](implicit broker: KernelBroker[T]): InPlaceImpl2[OpDiv.type, CuVector[T], T] =  broker.inPlaceImpl2For_v_s[OpDiv.type]("div")
  implicit def modIntoImpl_S[T](implicit broker: KernelBroker[T]): InPlaceImpl2[OpMod.type, CuVector[T], T] =  broker.inPlaceImpl2For_v_s[OpMod.type]("mod")
  implicit def maxIntoImpl_S[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImpl2For_v_s[max.type]("max")
  implicit def minIntoImpl_S[T](implicit broker: KernelBroker[T]): InPlaceImpl2[min.type, CuVector[T], T] =  broker.inPlaceImpl2For_v_s[min.type]("min")
  implicit def powIntoImpl_S[T](implicit broker: KernelBroker[T]) =  broker.inPlaceImpl2For_v_s[OpPow.type]("pow")
  implicit def setIntoImpl_S[T](implicit broker: KernelBroker[T]): InPlaceImpl2[OpSet.type, CuVector[T], T] =  broker.inPlaceImpl2For_v_s[OpSet.type]("set")

  implicit def addImplVS[T](implicit broker: KernelBroker[T]): UImpl2[OpAdd.type, CuVector[T], T, CuVector[T]] =  broker.impl2For_v_s[OpAdd.type]("add")
  implicit def subImplVS[T](implicit broker: KernelBroker[T]): UImpl2[OpSub.type, CuVector[T], T, CuVector[T]] =  broker.impl2For_v_s[OpSub.type]("sub")
  implicit def mulImplVS[T](implicit broker: KernelBroker[T]): UImpl2[OpMulScalar.type, CuVector[T], T, CuVector[T]] =  broker.impl2For_v_s[OpMulScalar.type]("mul")
  implicit def mulMatrixImplVS[T](implicit broker: KernelBroker[T]): UImpl2[OpMulMatrix.type, CuVector[T], T, CuVector[T]] =  broker.impl2For_v_s[OpMulMatrix.type]("mul")
  implicit def divImplVS[T](implicit broker: KernelBroker[T]): UImpl2[OpDiv.type, CuVector[T], T, CuVector[T]] =  broker.impl2For_v_s[OpDiv.type]("div")
  implicit def modImplVS[T](implicit broker: KernelBroker[T]): UImpl2[OpMod.type, CuVector[T], T, CuVector[T]] =  broker.impl2For_v_s[OpMod.type]("mod")
  implicit def powImplVS[T](implicit broker: KernelBroker[T]) =  broker.impl2For_v_s[OpPow.type]("pow")

  implicit def addImplSV[T](implicit broker: KernelBroker[T]) =  broker.impl2For_s_v[OpAdd.type]("add")
  implicit def subImplSV[T](implicit broker: KernelBroker[T]) =  broker.impl2For_s_v[OpSub.type]("sub")
  implicit def mulImplSV[T](implicit broker: KernelBroker[T]) =  broker.impl2For_s_v[OpMulScalar.type]("mul")
  implicit def mulMatrixImplSV[T](implicit broker: KernelBroker[T]) =  broker.impl2For_s_v[OpMulMatrix.type]("mul")
  implicit def divImplSV[T](implicit broker: KernelBroker[T]) =  broker.impl2For_s_v[OpDiv.type]("div")
  implicit def modImplSV[T](implicit broker: KernelBroker[T]) =  broker.impl2For_s_v[OpMod.type]("mod")
  implicit def powImplSV[T](implicit broker: KernelBroker[T]) =  broker.impl2For_s_v[OpPow.type]("pow")

  implicit def sumImpl[T](implicit broker: KernelBroker[T]) =  broker.reducerFor[sum.type]("add")
  implicit def maxReduceImpl[T](implicit broker: KernelBroker[T]) =  broker.reducerFor[max.type]("max")
  implicit def minReduceImpl[T](implicit broker: KernelBroker[T]) =  broker.reducerFor[min.type]("min")

  /*
  implicit def sumColImpl[T](implicit broker: KernelBroker[T]) =  broker.colReducerFor[sum.type]("add")
  implicit def maxColImpl[T](implicit broker: KernelBroker[T]) =  broker.colReducerFor[max.type]("max")
  implicit def minColImpl[T](implicit broker: KernelBroker[T]) =  broker.colReducerFor[min.type]("min")

  implicit def sumRowImpl[T](implicit broker: KernelBroker[T]) =  broker.rowReducerFor[sum.type]("add")
  implicit def maxRowImpl[T](implicit broker: KernelBroker[T]) =  broker.rowReducerFor[max.type]("max")
  implicit def minRowImpl[T](implicit broker: KernelBroker[T]) =  broker.rowReducerFor[min.type]("min")
  */

  class SetCuMCuMVOp[V] extends OpSet.InPlaceImpl2[CuVector[V], CuVector[V]] {
    def apply(a: CuVector[V], b: CuVector[V]) {
      a.writeFrom(b.asInstanceOf[CuVector[V]])
    }
  }

  implicit def setCuMCuMOp[V]:OpSet.InPlaceImpl2[CuVector[V], CuVector[V]] = new SetCuMCuMVOp[V]()

  implicit def setMDM[V](implicit stream: CUstream = new CUstream()): OpSet.InPlaceImpl2[CuVector[V], DenseVector[V]] = new OpSet.InPlaceImpl2[CuVector[V], DenseVector[V]] {
    def apply(v: CuVector[V], v2: DenseVector[V]): Unit = {
      v.writeFromDense(v2)
    }
  }

  implicit object softmaxImplFloat extends softmax.Impl[CuVector[Float], Float] {
    override def apply(v: CuVector[Float]): Float = {
      val m: Float = max(v)
      val temp = v - m
      exp.inPlace(temp)
      val res = log(sum(temp)) + m
      temp.data.release()
      res
    }
  }




}

trait CuVectorLowPrio { this: CuVector.type =>
  /** lbfgs wants a MIPS[T, Double], so this implicit allows us to fake it. */
  implicit def vspaceFloatPretendsToBeDouble(implicit handle: cublasHandle): MutableInnerProductVectorSpace[CuVector[Float], Double] = {

    implicit object addVSDouble extends OpAdd.Impl2[CuVector[Float], Double, CuVector[Float]] {
      override def apply(v: CuVector[Float], v2: Double): CuVector[Float] = v :+ v2.toFloat
    }

    implicit object subVSDouble extends OpSub.Impl2[CuVector[Float], Double, CuVector[Float]] {
      override def apply(v: CuVector[Float], v2: Double): CuVector[Float] = v :- v2.toFloat
    }

    implicit object mulVSDouble extends OpMulScalar.Impl2[CuVector[Float], Double, CuVector[Float]] {
      override def apply(v: CuVector[Float], v2: Double) = {v :* v2.toFloat}
    }

    implicit object addIntoVSDouble extends OpAdd.InPlaceImpl2[CuVector[Float], Double] {
      override def apply(v: CuVector[Float], v2: Double) = { v :+= v2.toFloat}
    }

    implicit object subIntoVSDouble extends OpSub.InPlaceImpl2[CuVector[Float], Double] {
      override def apply(v: CuVector[Float], v2: Double) = {v :-= v2.toFloat}
    }


    implicit object divVSDouble extends OpDiv.Impl2[CuVector[Float], Double, CuVector[Float]] {
      override def apply(v: CuVector[Float], v2: Double): CuVector[Float] = v :/ v2.toFloat
    }


    implicit object mulInner extends OpMulInner.Impl2[CuVector[Float], CuVector[Float], Double] {
      override def apply(v: CuVector[Float], v2: CuVector[Float]): Double = CuVector.canDotFloat(handle)(v, v2).toDouble
    }


    implicit object mulIntoVSDouble extends OpMulScalar.InPlaceImpl2[CuVector[Float], Double] {
      override def apply(v: CuVector[Float], v2: Double) = v :*= v2.toFloat
    }


    implicit object divIntoVSDouble extends OpDiv.InPlaceImpl2[CuVector[Float], Double] {
      override def apply(v: CuVector[Float], v2: Double) = v :/= v2.toFloat
    }

    implicit object scaleAddVS extends scaleAdd.InPlaceImpl3[CuVector[Float], Double, CuVector[Float]] {
        override def apply(v: CuVector[Float], v2: Double, v3: CuVector[Float]): Unit = {
          scaleAdd.inPlace(v, v2.toFloat, v3)
        }
    }

    MutableInnerProductVectorSpace.make[CuVector[Float], Double]



  }
}
