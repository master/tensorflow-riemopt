"""Manifold of linear subspaces of a vector space."""

import tensorflow as tf

from tensorflow_riemopt.manifolds.manifold import Manifold
from tensorflow_riemopt.manifolds import utils


class Grassmannian(Manifold):
    """Manifold of :math:`k`-dimensional linear subspaces in the
    :math:`n`-dimensional Euclidean space.

    Edelman, Alan, Tomás A. Arias, and Steven T. Smith. "The geometry of
    algorithms with orthogonality constraints." SIAM journal on Matrix
    Analysis and Applications 20.2 (1998): 303-353.
    """

    name = "Grassmannian"
    ndims = 2

    def _check_shape(self, shape):
        return shape[-2] >= shape[-1]

    def _check_point_on_manifold(self, x, atol, rtol):
        xtx = utils.transposem(x) @ x
        shape = xtx.shape.as_list()
        eye = tf.eye(shape[-1], batch_shape=shape[:-2])
        is_idempotent = utils.allclose(xtx, tf.cast(eye, x.dtype), atol, rtol)
        s = tf.linalg.svd(x, compute_uv=False)
        rank = tf.math.count_nonzero(s, axis=-1, dtype=tf.float32)
        k = tf.ones_like(rank) * int(x.shape[-1])
        is_col_rank = utils.allclose(rank, k, atol, rtol)
        return is_idempotent & is_col_rank

    def _check_vector_on_tangent(self, x, u, atol, rtol):
        xtu = utils.transposem(x) @ u
        return utils.allclose(xtu, tf.zeros_like(xtu), atol, rtol)

    def projx(self, x):
        q, _r = tf.linalg.qr(x)
        return q

    def proju(self, x, u):
        xtu = utils.transposem(x) @ u
        return u - x @ xtu

    def dist(self, x, y, keepdims=False):
        s = tf.linalg.svd(utils.transposem(x) @ y, compute_uv=False)
        theta = tf.math.acos(tf.clip_by_value(s, -1.0, 1.0))
        norm = tf.linalg.norm(theta, axis=-1, keepdims=False)
        return norm[..., tf.newaxis, tf.newaxis] if keepdims else norm

    def inner(self, x, u, v, keepdims=False):
        return tf.reduce_sum(u * v, axis=[-2, -1], keepdims=keepdims)

    def exp(self, x, u):
        s, u, v = tf.linalg.svd(u, full_matrices=False)
        cos_s = tf.linalg.diag(tf.math.cos(s))
        sin_s = tf.linalg.diag(tf.math.sin(s))
        return (x @ v @ cos_s + u @ sin_s) @ utils.transposem(v)

    def log(self, x, y):
        ytx = utils.transposem(y) @ x
        at = utils.transposem(y) - ytx @ utils.transposem(x)
        bt = tf.linalg.solve(ytx, at)
        s, u, v = tf.linalg.svd(utils.transposem(bt), full_matrices=False)
        return u @ tf.linalg.diag(tf.math.atan(s)) @ utils.transposem(v)

    def geodesic(self, x, u, t):
        s, u, v = tf.linalg.svd(u, full_matrices=False)
        cos_ts = tf.linalg.diag(tf.math.cos(t * s))
        sin_ts = tf.linalg.diag(tf.math.sin(t * s))
        return (x @ v @ cos_ts + u @ sin_ts) @ utils.transposem(v)

    def transp(self, x, y, v):
        return self.proju(y, v)

    def retr(self, x, u):
        _s, u, v = tf.linalg.svd(x + u, full_matrices=False)
        return u @ utils.transposem(v)

    def ptransp(self, x, y, v):
        log_xy = self.log(x, y)
        s, u, w = tf.linalg.svd(log_xy, full_matrices=False)
        cos_s = tf.linalg.diag(tf.math.cos(s))
        sin_s = tf.linalg.diag(tf.math.sin(s))
        geod = (-x @ w @ sin_s + u @ cos_s) @ utils.transposem(u) @ v
        proj = v - u @ utils.transposem(u) @ v
        return geod + proj
