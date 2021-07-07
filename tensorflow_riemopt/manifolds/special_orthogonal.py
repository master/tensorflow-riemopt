"""Manifold of rotation matrices."""
import tensorflow as tf

from tensorflow_riemopt.manifolds.manifold import Manifold
from tensorflow_riemopt.manifolds import utils


class SpecialOrthogonal(Manifold):
    """Manifold of square orthogonal matrices of determinant 1."""

    name = "Special Orthogonal"
    ndims = 2

    def _check_shape(self, shape):
        return shape[-2] == shape[-1]

    def _check_point_on_manifold(self, x, atol, rtol):
        xtx = utils.transposem(x) @ x
        eye = tf.eye(
            tf.shape(xtx)[-1], batch_shape=tf.shape(xtx)[:-2], dtype=x.dtype
        )
        is_orth = utils.allclose(xtx, eye, atol, rtol)
        det = tf.linalg.det(x)
        is_unit_det = utils.allclose(det, tf.ones_like(det), atol, rtol)
        return is_orth & is_unit_det

    def _check_vector_on_tangent(self, x, u, atol, rtol):
        diff = utils.transposem(u) + u
        return utils.allclose(diff, tf.zeros_like(diff), atol, rtol)

    def dist(self, x, y, keepdims=False):
        return tf.linalg.norm(self.log(x, y), axis=[-2, -1], keepdims=keepdims)

    def inner(self, x, u, v, keepdims=False):
        return tf.reduce_sum(u * v, axis=[-2, -1], keepdims=keepdims)

    def proju(self, x, u):
        xtu = utils.transposem(x) @ u
        return (xtu - utils.transposem(xtu)) / 2.0

    def projx(self, x):
        s, u, vt = tf.linalg.svd(x)
        ones = tf.ones_like(s)[..., :-1]
        signs = tf.cast(tf.sign(tf.linalg.det(u @ vt)), ones.dtype)
        flip = tf.concat([ones, tf.expand_dims(signs, -1)], axis=-1)
        return u @ tf.linalg.diag(flip) @ vt

    def exp(self, x, u):
        return x @ tf.linalg.expm(u)

    def retr(self, x, u):
        q, r = tf.linalg.qr(x + x @ u)
        unflip = tf.sign(tf.sign(tf.linalg.diag_part(r)) + 0.5)
        return q * unflip[..., tf.newaxis, :]

    def retr2(self, x, u):
        _s, u, vt = tf.linalg.svd(x + x @ u)
        return u @ vt

    def log(self, x, y):
        xty = utils.transposem(x) @ y
        u = utils.logm(xty)
        return (u - utils.transposem(u)) / 2.0

    def transp(self, x, y, v):
        return v

    ptransp = transp

    def pairmean(self, x, y):
        return self.exp(x, self.log(x, y) / 2.0)

    def geodesic(self, x, u, t):
        return x @ tf.linalg.expm(t * u)
