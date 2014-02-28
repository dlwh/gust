package snap.util

/**
 * TODO
 *
 * @author dlwh
 **/
trait CanRepresentAs[From, To] {
  def convert(v: From):To
  def unconvert(v: To):From
}

object CanRepresentAs extends CanRepresentAsLowPrio {
  implicit def canRepresentTasT[T]: CanRepresentAs[T, T] = new CanRepresentAs[T, T] {
    def convert(v: T): T = v
    def unconvert(v: T): T = v
  }
  implicit object FloatsIsInts extends CanRepresentAs[Float, Int] {
    def convert(v: Float): Int = java.lang.Float.floatToRawIntBits(v)
    def unconvert(v: Int): Float = java.lang.Float.intBitsToFloat(v)
  }

}

trait CanRepresentAsLowPrio {
  implicit def reverse[T, U](implicit cast: CanRepresentAs[T, U]):CanRepresentAs[U, T] = {
    new CanRepresentAs[U, T] {
      def convert(v: U): T = cast.unconvert(v)

      def unconvert(v: T): U = cast.convert(v)
    }
  }
}
