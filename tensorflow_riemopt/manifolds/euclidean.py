"""Manifold of the Euclidean space."""
import tensorflow as tf

from tensorflow_riemopt.manifolds.manifold import Manifold


class Euclidean(Manifold):
    name = "Euclidean"
    ndims = 0

    def __init__(self, ndims=0):
        """Instantiate the Euclidean manifold.

        Args:
          ndims: number of dimensions
        """
        super().__init__()
        self.ndims = ndims

    def _check_point_on_manifold(self, x, atol, rtol):
        return tf.constant(True)

    def _check_vector_on_tangent(self, x, u, atol, rtol):
        return tf.constant(True)

    def dist(self, x, y, keepdims=False):
        return self.norm(x, x - y, keepdims=keepdims)

    def inner(self, x, u, v, keepdims=False):
        return tf.reduce_sum(
            u * v, axis=tuple(range(-self.ndims, 0)), keepdims=keepdims
        )

    def proju(self, x, u):
        return u

    def projx(self, x):
        return x

    def exp(self, x, u):
        return x + u

    retr = exp

    def log(self, x, y):
        return y - x

    def ptransp(self, x, y, v):
        return v

    transp = ptransp

    def geodesic(self, x, u, t):
        return x + t * u

    def pairmean(self, x, y):
        return (x + y) / 2.0
