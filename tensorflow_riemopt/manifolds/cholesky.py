"""Manifold of the Cholesky space."""
import tensorflow as tf

from tensorflow_riemopt.manifolds.manifold import Manifold
from tensorflow_riemopt.manifolds import utils


class Cholesky(Manifold):
    """Manifold of lower triangular matrices with positive diagonal elements.

    Lin, Zhenhua. "Riemannian Geometry of Symmetric Positive Definite Matrices
    via Cholesky Decomposition." SIAM Journal on Matrix Analysis and
    Applications 40.4 (2019): 1353-1370.

    """

    name = "Cholesky"
    ndims = 2

    def _check_shape(self, shape):
        return shape[-1] == shape[-2]

    def _check_point_on_manifold(self, x, atol, rtol):
        lower_triang = tf.linalg.band_part(x, -1, 0)
        is_lower_triang = utils.allclose(x, lower_triang, atol, rtol)
        diag = tf.linalg.diag_part(x)
        is_pos_diag = utils.allclose(diag, tf.abs(diag), atol, rtol)
        return is_lower_triang & is_pos_diag

    def _check_vector_on_tangent(self, x, u, atol, rtol):
        lower_triang = tf.linalg.band_part(x, -1, 0)
        return utils.allclose(x, lower_triang, atol, rtol)

    def _diag_and_strictly_lower(self, x):
        x = tf.linalg.band_part(x, -1, 0)
        diag = tf.linalg.diag_part(x)
        return diag, x - tf.linalg.diag(diag)

    def projx(self, x):
        x_sym = (utils.transposem(x) + x) / 2.0
        s, _u, v = tf.linalg.svd(x_sym)
        sigma = tf.linalg.diag(tf.maximum(s, 0.0))
        spd = v @ sigma @ utils.transposem(v)
        return tf.linalg.cholesky(spd)

    def proju(self, x, u):
        u_sym = (utils.transposem(u) + u) / 2.0
        u_diag, u_lower = self._diag_and_strictly_lower(u_sym)
        x_diag = tf.linalg.diag_part(x)
        return u_lower + tf.linalg.diag(u_diag * x_diag**2)

    def inner(self, x, u, v, keepdims=False):
        u_diag, u_lower = self._diag_and_strictly_lower(u)
        v_diag, v_lower = self._diag_and_strictly_lower(v)
        x_diag = tf.linalg.diag_part(x)
        lower = tf.reduce_sum(
            u_lower * v_lower, axis=[-2, -1], keepdims=keepdims
        )
        diag = tf.reduce_sum(
            u_diag * v_diag / tf.math.square(x_diag), axis=[-1]
        )
        return lower + tf.reshape(diag, lower.shape)

    def geodesic(self, x, u, t):
        x_diag, x_lower = self._diag_and_strictly_lower(x)
        u_diag, u_lower = self._diag_and_strictly_lower(u)
        diag = tf.linalg.diag(x_diag * tf.math.exp(t * u_diag / x_diag))
        return x_lower + t * u_lower + diag

    def exp(self, x, u):
        x_diag, x_lower = self._diag_and_strictly_lower(x)
        u_diag, u_lower = self._diag_and_strictly_lower(u)
        diag = tf.linalg.diag(x_diag * tf.math.exp(u_diag / x_diag))
        return x_lower + u_lower + diag

    retr = exp

    def log(self, x, y):
        x_diag, x_lower = self._diag_and_strictly_lower(x)
        y_diag, y_lower = self._diag_and_strictly_lower(y)
        diag = tf.linalg.diag(x_diag * tf.math.log(y_diag / x_diag))
        return y_lower - x_lower + diag

    def dist(self, x, y, keepdims=False):
        x_diag, x_lower = self._diag_and_strictly_lower(x)
        y_diag, y_lower = self._diag_and_strictly_lower(y)
        lower = tf.linalg.norm(
            x_lower - y_lower, axis=[-2, -1], ord="fro", keepdims=keepdims
        )
        diag = tf.linalg.norm(
            tf.math.log(x_diag) - tf.math.log(y_diag),
            axis=-1,
            keepdims=keepdims,
        )
        return tf.math.sqrt(lower**2 + tf.reshape(diag, lower.shape) ** 2)

    def ptransp(self, x, y, v):
        x_diag, _ = self._diag_and_strictly_lower(x)
        y_diag, _ = self._diag_and_strictly_lower(y)
        v_diag, v_lower = self._diag_and_strictly_lower(v)
        diag = tf.linalg.diag(v_diag * y_diag / x_diag)
        return v_lower + diag

    transp = ptransp

    def pairmean(self, x, y):
        return self.geodesic(x, self.log(x, y), 0.5)
