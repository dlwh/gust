package gust.linalg.cuda

import org.scalatest.FunSuite
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import breeze.math.MutableVectorSpace
import org.scalatest.prop.Checkers
import jcuda.jcublas.{JCublas2, cublasHandle}
import org.scalacheck._
import breeze.linalg._

/**
 * TODO
 *
 * @author dlwh
 **/
class CuVectorTest extends FunSuite {

  test("Simple dot") {
    implicit val handle = new cublasHandle
    JCublas2.cublasCreate(handle)
    val a = CuVector.rand(10)
    val ad = a.toDense.mapValues(_.toDouble)
    assert((a.dot(a) - ad.dot(ad)).abs < 1E-4)

    JCublas2.cublasDestroy(handle)
  }

  test("max") {
    val rand = convert(DenseVector.rand(40), Float)
    val cumat = CuVector.zeros[Float](40)
    cumat := rand
    assert(max(cumat) === max(rand))
  }

  test("softmax") {
    val rand = convert(DenseVector.rand(40), Float)
    val cumat = CuVector.zeros[Float](40)
    cumat := rand
    val cumax = softmax(cumat)
    val dmax = softmax(convert(rand, Double))
    assert(math.abs(cumax - dmax) < 1e-4)
  }

}

@RunWith(classOf[JUnitRunner])
class CuVectorOps_FloatTest extends MutableVectorSpaceTestBase[CuVector[Float], Float] {
  implicit val handle = new cublasHandle
  JCublas2.cublasCreate(handle)
  val space: MutableVectorSpace[CuVector[Float], Float] = implicitly

  val N = 30
  implicit def genTriple: Arbitrary[(CuVector[Float], CuVector[Float], CuVector[Float])] = {
    Arbitrary {
      for{x <- Arbitrary.arbitrary[Float].map { _  % 1000} if !x.isInfinite
          y <- Arbitrary.arbitrary[Float].map { _ % 1000} if !y.isInfinite
          z <- Arbitrary.arbitrary[Float].map { _ % 1000} if !z.isInfinite
      } yield {
        (CuVector.fromDense(DenseVector.rand(N).mapValues(_.toFloat)) *= x,
          CuVector.fromDense(DenseVector.rand(N).mapValues(_.toFloat)) *= y,
            CuVector.fromDense(DenseVector.rand(N).mapValues(_.toFloat)) *= z)
      }
    }
  }

  def genScalar: Arbitrary[Float] = Arbitrary(Arbitrary.arbitrary[Float].map{ _ % 1000 })
}

trait MutableVectorSpaceTestBase[V, S] extends FunSuite with Checkers {
  implicit val space: MutableVectorSpace[V,  S]
  import space._


  implicit def genTriple: Arbitrary[(V, V, V)]
  implicit def genScalar: Arbitrary[S]

  val TOL = 1E-4

  test("Addition is Associative") {
     check(Prop.forAll{ (trip: (V, V, V)) =>
       val (a, b, c) = trip
       val abc = (a + b) + c
       val a_bc = a + (b + c)
       val res = close(abc, a_bc, TOL)
       if(!res) {
         println(a.asInstanceOf[CuVector[Float]].toDense)
         println(b.asInstanceOf[CuVector[Float]].toDense)
         println(c.asInstanceOf[CuVector[Float]].toDense)
         println(abc.asInstanceOf[CuVector[Float]].toDense)
         println(a_bc.asInstanceOf[CuVector[Float]].toDense)
       }
       res
     })

     check(Prop.forAll{ (trip: (V, V, V)) =>
       val (a, b, c) = trip
       val ab = a + b
       val bc = b + c
       ab += c
       bc += a
       close(ab, bc, TOL)
     })
   }

   test("Addition Commutes") {
     check(Prop.forAll{ (trip: (V, V, V)) =>
       val (a, b, _) = trip
       close(a + b, b +a, TOL)
     })

     check(Prop.forAll{ (trip: (V, V, V)) =>
       val (a, b, _) = trip
       val ab = copy(a)
       ab += b
       val ba = copy(b)
       ba += a
       close(ab, ba, TOL)
     })
   }

   test("Zero is Zero") {
     check(Prop.forAll{ (trip: (V, V, V)) =>
       val (a, b, c) = trip
       val z = zeros(a)
       close(a :+ z, a, TOL)
     })

     check(Prop.forAll{ (trip: (V, V, V)) =>
       val (a, b, _) = trip
       val ab = copy(a)
       val z = zeros(a)
       ab :+= z
       close(a, ab, TOL)
     })
   }

   test("a + -a == 0") {
     check(Prop.forAll{ (trip: (V, V, V)) =>
       val (a, b, c) = trip
       val z = zeros(a)
       close(a + -a, z, TOL)
     })

     check(Prop.forAll{ (trip: (V, V, V)) =>
       val (a, b, _) = trip
       val z = zeros(a)
       a += -a
       close(a, z, TOL)
     })

     check(Prop.forAll{ (trip: (V, V, V)) =>
       val (a, b, _) = trip
       val z = zeros(a)
       a :-= a
       close(a, z, TOL)
     })

     check(Prop.forAll{ (trip: (V, V, V)) =>
       val (a, b, _) = trip
       val z = zeros(a)
       val ab = a :- b
       a -= b
       close(a, ab, TOL)
     })
   }

   test("Scalar mult distributes over vector addition") {
     check(Prop.forAll{ (trip: (V, V, V), s: S) =>
       val (a, b, _) = trip
       close( (a + b) * s, (b * s +a * s), TOL)
     })

 //    check(Prop.forAll{ (trip: (V, V, V), s: S) =>
 //      val (a, b, _) = trip
 //      s == 0 || close( (a + b)/ s, (b / s +a / s), TOL)
 //    })

     check(Prop.forAll{ (trip: (V, V, V), s: S) =>
       val (a, b, _) = trip
       val ab = copy(a)
       ab += b
       ab *= s
       val ba = copy(a) * s
       ba += (b * s)
       close(ab, ba, TOL)
     })



   }

  test("daxpy is consistent") {
    check(Prop.forAll{ (trip: (V, V, V), s: S) =>
      val (a, b, _) = trip
      val ac = copy(a)
      val prod = a + b * s
      breeze.linalg.axpy(s, b, ac)
      close( prod, ac, TOL)
    })

  }


   test("Scalar mult distributes over field addition") {
     check(Prop.forAll{ (trip: (V, V, V), s: S, t: S) =>
       val (a, _, _) = trip
       close( (a) * field.+(s,t), (a * s + a * t), 1E-4)
     })

     check(Prop.forAll{ (trip: (V, V, V), s: S, t: S) =>
       val (a, _, _) = trip
       val ab = copy(a)
       ab *= s
       ab += (a * t)
       val ba = copy(a)
       ba *= field.+(s,t)
       close(ab, ba, 1e-4)
     })
   }

   test("Compatibility of scalar multiplication with field multiplication") {
     check(Prop.forAll{ (trip: (V, V, V), s: S, t: S) =>
       val (a, _, _) = trip
       close( (a) * field.*(s,t), a * s * t, TOL)
     })

     check(Prop.forAll{ (trip: (V, V, V), s: S, t: S) =>
       val (a, _, _) = trip
       val ab = copy(a)
       ab *= s
       ab *= t
       val ba = copy(a)
       ba *= field.*(s, t)
       close(ab, ba, TOL)
     })

     check(Prop.forAll{ (trip: (V, V, V), s: S, t: S) =>
       val (a, _, _) = trip
       s == field.zero || t == field.zero || {
       val ab = copy(a)
         ab /= s
         ab /= t
         val ba = copy(a)
         ba /= field.*(s, t)
         close(ab, ba, TOL)
       }
     })
   }

  // op set
  test("op set works") {
    check(Prop.forAll{ (trip: (V, V, V)) =>
      val (a, b, _) = trip
      val ab = copy(a)
      ab := b
      val apv = a + b
      val apab = a + ab
      val res = close(apv, apab, 1E-10)
      res
    })

  }


  test("1 is 1") {
    check(Prop.forAll{ (trip: (V, V, V)) =>
      val (a, b, c) = trip
      close(a * field.one, a, TOL)
    })

    check(Prop.forAll{ (trip: (V, V, V)) =>
      val (a, b, _) = trip
      val ab = copy(a)
      ab *= field.one
      close(a, ab, TOL)
    })
  }
}