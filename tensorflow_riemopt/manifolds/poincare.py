"""The Poincaré ball model."""
import tensorflow as tf

from tensorflow_riemopt.manifolds.manifold import Manifold
from tensorflow_riemopt.manifolds import utils


class Poincare(Manifold):
    """Manifold of `math:`n`-dimensional hyperbolic space as embedded in the
    Poincaré ball model.

    Nickel, Maximillian, and Douwe Kiela. "Poincaré embeddings for learning
    hierarchical representations." Advances in neural information processing
    systems. 2017.

    Ganea, Octavian, Gary Bécigneul, and Thomas Hofmann. "Poincaré neural
    networks." Advances in neural information processing systems. 2018.

    """

    name = "Poincaré"
    ndims = 1

    def __init__(self, k=1.0):
        """Instantiate the Poincaré manifold.

        Args:
          k: scale of the hyperbolic space, k > 0.
        """
        self.k = k
        super().__init__()

    def __repr__(self):
        return "{0} (k={1}, ndims={2}) manifold".format(
            self.name, self.k, self.ndims
        )

    def _check_point_on_manifold(self, x, atol, rtol):
        k = tf.cast(self.k, x.dtype)
        sq_norm = tf.reduce_sum(x * x, axis=-1, keepdims=True)
        return tf.reduce_all(sq_norm * k < tf.ones_like(sq_norm))

    def _check_vector_on_tangent(self, x, u, atol, rtol):
        return tf.constant(True)

    def _mobius_add(self, x, y):
        """Compute the Möbius addition of :math:`x` and :math:`y` in
        :math:`\mathcal{D}^{n}_{k}`

        :math:`x \oplus y = \frac{(1 + 2k\langle x, y\rangle + k||y||^2)x + (1
            - k||x||^2)y}{1 + 2k\langle x,y\rangle + k^2||x||^2||y||^2}`
        """
        x_2 = tf.reduce_sum(tf.math.square(x), axis=-1, keepdims=True)
        y_2 = tf.reduce_sum(tf.math.square(y), axis=-1, keepdims=True)
        x_y = tf.reduce_sum(x * y, axis=-1, keepdims=True)
        k = tf.cast(self.k, x.dtype)
        return ((1 + 2 * k * x_y + k * y_2) * x + (1 - k * x_2) * y) / (
            1 + 2 * k * x_y + k**2 * x_2 * y_2
        )

    def _mobius_scal_mul(self, x, r):
        """Compute the Möbius scalar multiplication of :math:`x \in
        \mathcal{D}^{n}_{k} \ {0}` by :math:`r`

        :math:`x \otimes r = (1/\sqrt{k})\tanh(r
        \atanh(\sqrt{k}||x||))\frac{x}{||x||}`

        """
        sqrt_k = tf.math.sqrt(tf.cast(self.k, x.dtype))
        norm_x = tf.linalg.norm(x, axis=-1, keepdims=True)
        eps = utils.get_eps(x)
        tan = tf.clip_by_value(sqrt_k * norm_x, -1.0 + eps, 1.0 - eps)
        return (1 / sqrt_k) * tf.math.tanh(r * tf.math.atanh(tan)) * x / norm_x

    def _gyration(self, u, v, w):
        """Compute the gyration of :math:`u`, :math:`v`, :math:`w`:

        :math:`\operatorname{gyr}[u, v]w =
        \ominus (u \oplus_\kappa v) \oplus (u \oplus_\kappa (v \oplus_\kappa w))`
        """
        min_u_v = -self._mobius_add(u, v)
        v_w = self._mobius_add(v, w)
        u_v_w = self._mobius_add(u, v_w)
        return self._mobius_add(min_u_v, u_v_w)

    def _lambda(self, x, keepdims=False):
        """Compute the conformal factor :math:`lambda_x^k`"""
        k = tf.cast(self.k, x.dtype)
        norm_x_2 = tf.reduce_sum(x * x, axis=-1, keepdims=keepdims)
        return 2.0 / (1.0 - k * norm_x_2)

    def inner(self, x, u, v, keepdims=False):
        lambda_x = self._lambda(x, keepdims=keepdims)
        return tf.reduce_sum(u * v, axis=-1, keepdims=keepdims) * lambda_x**2

    def norm(self, x, u, keepdims=False):
        lambda_x = self._lambda(x, keepdims=keepdims)
        return tf.linalg.norm(u, axis=-1, keepdims=keepdims) * lambda_x

    def proju(self, x, u):
        lambda_x = self._lambda(x, keepdims=True)
        return u / lambda_x**2

    def projx(self, x):
        sqrt_k = tf.math.sqrt(tf.cast(self.k, x.dtype))
        norm = tf.linalg.norm(x, axis=-1, keepdims=True)
        return tf.where(
            sqrt_k * norm < tf.ones_like(norm),
            x,
            x / (sqrt_k * norm + 10 * utils.get_eps(x)),
        )

    def dist(self, x, y, keepdims=False):
        sqrt_k = tf.math.sqrt(tf.cast(self.k, x.dtype))
        x_y = self._mobius_add(-x, y)
        norm_x_y = tf.linalg.norm(x_y, axis=-1, keepdims=keepdims)
        eps = utils.get_eps(x)
        tanh = tf.clip_by_value(sqrt_k * norm_x_y, -1.0 + eps, 1.0 - eps)
        return 2 * tf.math.atanh(tanh) / sqrt_k

    def exp(self, x, u):
        sqrt_k = tf.math.sqrt(tf.cast(self.k, x.dtype))
        norm_u = tf.linalg.norm(u, axis=-1, keepdims=True)
        lambda_x = self._lambda(x, keepdims=True)
        y = (
            tf.math.tanh(sqrt_k * norm_u * lambda_x / 2.0)
            * u
            / (sqrt_k * norm_u)
        )
        return self._mobius_add(x, y)

    def log(self, x, y):
        sqrt_k = tf.math.sqrt(tf.cast(self.k, x.dtype))
        x_y = self._mobius_add(-x, y)
        norm_x_y = tf.linalg.norm(x_y, axis=-1, keepdims=True)
        eps = utils.get_eps(x)
        tanh = tf.clip_by_value(sqrt_k * norm_x_y, -1.0 + eps, 1.0 - eps)
        lambda_x = self._lambda(x, keepdims=True)
        return 2 * (x_y / norm_x_y) * tf.math.atanh(tanh) / (sqrt_k * lambda_x)

    retr = exp

    def ptransp(self, x, y, v):
        lambda_x = self._lambda(x, keepdims=True)
        lambda_y = self._lambda(y, keepdims=True)
        return self._gyration(y, -x, v) * lambda_x / lambda_y

    transp = ptransp

    def geodesic(self, x, u, t):
        sqrt_k = tf.math.sqrt(tf.cast(self.k, x.dtype))
        norm_u = tf.linalg.norm(u, axis=-1, keepdims=True)
        y = tf.math.tanh(sqrt_k * t / 2.0) * u / (norm_u * sqrt_k)
        return self._mobius_add(x, y)

    def exp0(self, u):
        """Perform an exponential map from the origin"""
        sqrt_k = tf.math.sqrt(tf.cast(self.k, u.dtype))
        norm_u = tf.linalg.norm(u, axis=-1, keepdims=True)
        return tf.math.tanh(sqrt_k * norm_u) * u / (sqrt_k * norm_u)

    def log0(self, y):
        """Perform a logarithmic map from the origin"""
        sqrt_k = tf.math.sqrt(tf.cast(self.k, y.dtype))
        norm_y = tf.linalg.norm(y, axis=-1, keepdims=True)
        return tf.math.atanh(sqrt_k * norm_y) * y / (sqrt_k * norm_y)

    def ptransp0(self, y, v):
        """Perform a parallel transport from the origin"""
        lambda_y = self._lambda(y, keepdims=True)
        return 2 * v / lambda_y

    def random(self, shape, dtype=tf.float32):
        return self.projx(
            tf.random.uniform(shape, minval=-1e-3, maxval=1e-3, dtype=dtype)
        )
