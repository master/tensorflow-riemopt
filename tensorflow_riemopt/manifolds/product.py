"""Cartesian product of manifolds."""
import tensorflow as tf
from functools import reduce
from operator import mul

from tensorflow_riemopt.manifolds.manifold import Manifold


class Product(Manifold):
    """Product space of manifolds."""

    name = "Product"
    ndims = 1

    def __init__(self, *manifolds):
        """Initialize a product of manifolds.

        Args:
          *manifolds: an iterable of (`manifold`, `shape`) tuples, where
            `manifold` is an instance of `Manifold` and `shape` is a tuple of
            `manifold` dimensions

        Example:

        >>> from tensorflow_riemopt import manifolds

        >>> S = manifolds.Sphere()
        >>> torus = manifolds.Product((S, (2,)), (S, (2,)))

        >>> St = manifolds.EuclideanStiefel()
        >>> St_2 = manifolds.Product((St, (5, 3)), (St, (5, 3)))
        """
        self._indices = [0]
        self._manifolds = []
        self._shapes = []
        for i, (m, shape) in enumerate(manifolds):
            if not isinstance(m, Manifold):
                raise ValueError(
                    "{} should be an instance of Manifold".format(m)
                )
            if not m.check_shape(shape):
                raise ValueError(
                    "Invalid shape {} for manifold {}".format(shape, m)
                )
            self._indices.append(self._indices[i] + reduce(mul, shape))
            self._manifolds.append(m)
            self._shapes.append(list(shape))
        super().__init__()

    def __repr__(self):
        names = [
            "{}{}".format(m.name, tuple(shape))
            for m, shape in zip(self._manifolds, self._shapes)
        ]
        return " × ".join(names)

    def _check_shape(self, shape):
        return self._indices[-1] == shape[-1]

    def _check_point_on_manifold(self, x, atol, rtol):
        checks = [
            m.check_point_on_manifold(self._get_slice(x, i), atol, rtol)
            for (i, m) in enumerate(self._manifolds)
        ]
        return reduce(tf.logical_and, checks)

    def _check_vector_on_tangent(self, x, u, atol, rtol):
        checks = [
            m.check_vector_on_tangent(
                self._get_slice(x, i), self._get_slice(u, i), atol, rtol
            )
            for (i, m) in enumerate(self._manifolds)
        ]
        return reduce(tf.logical_and, checks)

    def _get_slice(self, x, idx):
        if not 0 <= idx < len(self._indices) - 1:
            raise ValueError("Invalid index {}".format(idx))
        slice = x[..., self._indices[idx] : self._indices[idx + 1]]
        shape = tf.concat([tf.shape(x)[:-1], self._shapes[idx]], axis=-1)
        return tf.reshape(slice, shape)

    def _product_fn(self, fn, *args, **kwargs):
        results = []
        for (i, m) in enumerate(self._manifolds):
            arg_slices = [self._get_slice(arg, i) for arg in args]
            result = getattr(m, fn)(*arg_slices, **kwargs)
            shape = tf.concat([tf.shape(result)[:-1], [-1]], axis=-1)
            results.append(tf.reshape(result, shape))
        return tf.concat(results, axis=-1)

    def random(self, shape, dtype=tf.float32):
        if not self.check_shape(shape):
            raise ValueError("Invalid shape {}".format(shape))
        shape = list(shape)
        results = []
        for (i, m) in enumerate(self._manifolds):
            result = m.random(shape[:-1] + self._shapes[i], dtype=dtype)
            results.append(tf.reshape(result, shape[:-1] + [-1]))
        return tf.concat(results, axis=-1)

    def dist(self, x, y, keepdims=False):
        dists = self._product_fn("dist", x, y, keepdims=True)
        shape = tf.concat([tf.shape(x)[:-1], [-1]], axis=-1)
        sq_dists = tf.reduce_sum(dists * dists, axis=-1, keepdims=keepdims)
        return tf.math.sqrt(sq_dists)

    def inner(self, x, u, v, keepdims=False):
        inners = self._product_fn("inner", x, u, v, keepdims=True)
        return tf.reduce_sum(inners, axis=-1, keepdims=keepdims)

    def proju(self, x, u):
        return self._product_fn("proju", x, u)

    def projx(self, x):
        return self._product_fn("projx", x)

    def exp(self, x, u):
        return self._product_fn("exp", x, u)

    def retr(self, x, u):
        return self._product_fn("retr", x, u)

    def log(self, x, y):
        return self._product_fn("log", x, y)

    def ptransp(self, x, y, v):
        return self._product_fn("ptransp", x, y, v)

    def transp(self, x, y, v):
        return self._product_fn("transp", x, y, v)

    def pairmean(self, x, y):
        return self._product_fn("pairmean", x, y)

    def geodesic(self, x, u, t):
        return self._product_fn("geodesic", x, u, t=t)

    def pairmean(self, x, y):
        return self._product_fn("pairmean", x, y)
