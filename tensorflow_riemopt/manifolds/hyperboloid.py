"""The Lorentz model."""
import tensorflow as tf

from tensorflow_riemopt.manifolds.manifold import Manifold
from tensorflow_riemopt.manifolds import utils


class Hyperboloid(Manifold):
    """Manifold of `math:`n`-dimensional hyperbolic space as embedded in
    :math:`n+1`-dimensional Minkowski space, also known as the Lorentz model.

    """

    name = "Hyperboloid"
    ndims = 1

    def __init__(self, k=1.0):
        """Instantiate the Hyperboloid manifold.

        Args:
          k: scale of the hyperbolic space.
        """
        self.k = k
        super().__init__()

    def __repr__(self):
        return "{0} (k={1}, ndims={2}) manifold".format(
            self.name, self.k, self.ndims
        )

    def _check_point_on_manifold(self, x, atol, rtol):
        x_sq = tf.square(x)
        quad_form = -x_sq[..., :1] + tf.reduce_sum(
            x_sq[..., 1:], axis=-1, keepdims=True
        )
        return utils.allclose(
            quad_form, tf.ones_like(quad_form) * -self.k, atol, rtol
        )

    def _check_vector_on_tangent(self, x, u, atol, rtol):
        inner = self.inner(x, x, u)
        rtol = 100 * utils.get_eps(x) if rtol is None else rtol
        return utils.allclose(inner, tf.zeros_like(inner), atol, rtol)

    def dist(self, x, y, keepdims=False):
        d = -self.inner(x, x, y, keepdims=keepdims)
        k = tf.cast(self.k, x.dtype)
        return tf.math.sqrt(k) * tf.math.acosh(tf.maximum(d / k, 1.0))

    def inner(self, x, u, v, keepdims=False):
        uv = u * v
        x0 = -tf.reduce_sum(uv[..., :1], axis=-1, keepdims=keepdims)
        return x0 + tf.reduce_sum(uv[..., 1:], axis=-1, keepdims=keepdims)

    def norm(self, x, u, keepdims=False):
        inner = self.inner(x, u, u, keepdims=keepdims)
        return tf.math.sqrt(tf.maximum(inner, utils.get_eps(x)))

    def proju(self, x, u):
        k = tf.cast(self.k, x.dtype)
        return u + self.inner(x, x, u, keepdims=True) * x / k

    def projx(self, x):
        k = tf.cast(self.k, x.dtype)
        x0 = tf.math.sqrt(
            k + tf.linalg.norm(x[..., 1:], axis=-1, keepdims=True) ** 2
        )
        return tf.concat([x0, x[..., 1:]], axis=-1)

    def log(self, x, y):
        k = tf.cast(self.k, x.dtype)
        dist = self.dist(x, y, keepdims=True)
        inner = self.inner(x, x, y, keepdims=True)
        p = y + 1.0 / k * inner * x
        return dist * p / self.norm(x, p, keepdims=True)

    def exp(self, x, u):
        sqrt_k = tf.math.sqrt(tf.cast(self.k, x.dtype))
        norm_u = self.norm(x, u, keepdims=True)
        return (
            tf.math.cosh(norm_u / sqrt_k) * x
            + sqrt_k * tf.math.sinh(norm_u / sqrt_k) * u / norm_u
        )

    retr = exp

    def ptransp(self, x, y, v):
        log_xy = self.log(x, y)
        log_yx = self.log(y, x)
        dist_sq = self.dist(x, y, keepdims=True) ** 2
        inner = self.inner(x, log_xy, v, keepdims=True)
        return v - inner * (log_xy + log_yx) / dist_sq

    transp = ptransp

    def to_poincare(self, x, k):
        """Diffeomorphism that maps to the Poincaré ball"""
        k = tf.cast(k, x.dtype)
        return x[..., 1:] / (x[..., :1] + tf.math.sqrt(k))

    def from_poincare(self, x, k):
        """Inverse of the diffeomorphism to the Poincaré ball"""
        k = tf.cast(k, x.dtype)
        x_sq_norm = tf.reduce_sum(x * x, axis=-1, keepdims=True)
        y = tf.math.sqrt(k) * tf.concat([1 + x_sq_norm, 2 * x], axis=-1)
        return y / (1.0 - x_sq_norm + utils.get_eps(x))
