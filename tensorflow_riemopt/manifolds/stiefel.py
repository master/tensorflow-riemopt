import tensorflow as tf

from tensorflow_riemopt.manifolds.manifold import Manifold
from tensorflow_riemopt.manifolds import utils


def blockm(a, b, c, d):
    """Form a block matrix.

    Returns a `Tensor`

    | a b |
    | c d |

    of shape [..., n, p], where n = n_1 + n_2, p = p_1 + p_2, and arguments
    have shapes

    `a` - [..., n_1, p_1]
    `b` - [..., n_1, p_2]
    `c` - [..., n_2, p_1]
    `d` - [..., n_2, p_2]

    """
    a_b = tf.concat([a, b], axis=-1)
    c_d = tf.concat([c, d], axis=-1)
    return tf.concat([a_b, c_d], axis=-2)


class _Stiefel(Manifold):
    """Manifold of orthonormal p-frames in the n-dimensional Euclidean space."""

    ndims = 2

    def _check_shape(self, shape):
        return shape[-2] >= shape[-1]

    def _check_point_on_manifold(self, x, atol, rtol):
        xtx = utils.transposem(x) @ x
        eye = tf.eye(
            tf.shape(xtx)[-1], batch_shape=tf.shape(xtx)[:-2], dtype=xtx.dtype
        )
        return utils.allclose(xtx, eye, atol, rtol)

    def _check_vector_on_tangent(self, x, u, atol, rtol):
        diff = utils.transposem(u) @ x + utils.transposem(x) @ u
        return utils.allclose(diff, tf.zeros_like(diff), atol, rtol)

    def projx(self, x):
        _s, u, vt = tf.linalg.svd(x)
        return u @ vt


class StiefelEuclidean(_Stiefel):
    """Manifold of orthonormal p-frames in the n-dimensional space endowed with
    the Euclidean inner product.

    """

    name = "Euclidean Stiefel"

    def inner(self, x, u, v, keepdims=False):
        return tf.reduce_sum(u * v, axis=[-2, -1], keepdims=keepdims)

    def proju(self, x, u):
        xtu = utils.transposem(x) @ u
        xtu_sym = (utils.transposem(xtu) + xtu) / 2.0
        return u - x @ xtu_sym

    def exp(self, x, u):
        return self.geodesic(x, u, 1.0)

    def retr(self, x, u):
        q, r = tf.linalg.qr(x + u)
        unflip = tf.cast(tf.sign(tf.linalg.diag_part(r)), r.dtype)
        return q * unflip[..., tf.newaxis, :]

    def geodesic(self, x, u, t):
        xtu = utils.transposem(x) @ u
        utu = utils.transposem(u) @ u
        eye = tf.eye(
            tf.shape(utu)[-1], batch_shape=tf.shape(utu)[:-2], dtype=x.dtype
        )
        logw = blockm(xtu, -utu, eye, xtu)
        w = tf.linalg.expm(t * logw)
        z = tf.concat([tf.linalg.expm(-xtu * t), tf.zeros_like(utu)], axis=-2)
        y = tf.concat([x, u], axis=-1) @ w @ z
        return y

    def dist(self, x, y, keepdims=False):
        raise NotImplementedError

    def log(self, x, y):
        return NotImplementedError

    def transp(self, x, y, v):
        return self.proju(y, v)


class StiefelCanonical(_Stiefel):
    """Manifold of orthonormal p-frames in the n-dimensional space endowed with
    the canonical inner product derived from the quotient space
    representation.

    Zimmermann, Ralf. "A matrix-algebraic algorithm for the Riemannian
    logarithm on the Stiefel manifold under the canonical metric." SIAM
    Journal on Matrix Analysis and Applications 38.2 (2017): 322-342.

    """

    name = "Canonical Stiefel"

    def inner(self, x, u, v, keepdims=False):
        xtu = utils.transposem(x) @ u
        xtv = utils.transposem(x) @ v
        u_v_inner = tf.reduce_sum(u * v, axis=[-2, -1], keepdims=keepdims)
        xtu_xtv_inner = tf.reduce_sum(
            xtu * xtv, axis=[-2, -1], keepdims=keepdims
        )
        return u_v_inner - 0.5 * xtu_xtv_inner

    def proju(self, x, u):
        return u - x @ utils.transposem(u) @ x

    def exp(self, x, u):
        return self.geodesic(x, u, 1.0)

    def retr(self, x, u):
        xut = x @ utils.transposem(u)
        xut_sym = (xut - utils.transposem(xut)) / 2.0
        eye = tf.eye(
            tf.shape(xut)[-1], batch_shape=tf.shape(xut)[:-2], dtype=x.dtype
        )
        return tf.linalg.solve(xut_sym + eye, x - xut_sym @ x)

    def geodesic(self, x, u, t):
        xtu = utils.transposem(x) @ u
        utu = utils.transposem(u) @ u
        eye = tf.eye(
            tf.shape(utu)[-1], batch_shape=tf.shape(utu)[:-2], dtype=x.dtype
        )
        q, r = tf.linalg.qr(u - x @ xtu)
        logw = blockm(xtu, -utils.transposem(r), r, tf.zeros_like(r))
        w = tf.linalg.expm(t * logw)
        z = tf.concat([eye, tf.zeros_like(utu)], axis=-2)
        y = tf.concat([x, u], axis=-1) @ w @ z
        return y

    def transp(self, x, y, v):
        return self.proju(y, v)

    def dist(self, x, y, keepdims=False):
        raise NotImplementedError

    def log(self, x, y):
        return NotImplementedError


class StiefelCayley(_Stiefel):
    """Manifold of orthonormal p-frames in the n-dimensional space endowed with
    the Euclidean inner product.

    Retraction and parallel transport operations are implemented using
    the iterative Cayley transform.

    Li, Jun, Fuxin Li, and Sinisa Todorovic. "Efficient Riemannian
    Optimization on the Stiefel Manifold via the Cayley Transform."
    International Conference on Learning Representations. 2019.

    """

    name = "Cayley Stiefel"

    def __init__(self, num_iter=10):
        """Instantiate the Stiefel manifold with Cayley transform.

        Args:
          num_iter: number of Cayley iterations to perform.
        """
        self.num_iter = num_iter
        super().__init__()

    def inner(self, x, u, v, keepdims=False):
        return tf.reduce_sum(u * v, axis=[-2, -1], keepdims=keepdims)

    def proju(self, x, u):
        xtu = utils.transposem(x) @ u
        w = (u - x @ xtu) @ utils.transposem(x)
        return (w - utils.transposem(w)) @ x

    def exp(self, x, u):
        return self.geodesic(x, u, 1.0)

    def geodesic(self, x, u, t):
        xtu = utils.transposem(x) @ u
        w_ = (u - x @ xtu) @ utils.transposem(x)
        w = w_ - utils.transposem(w_)
        eye = tf.linalg.eye(
            tf.shape(w)[-1], batch_shape=tf.shape(w)[:-2], dtype=x.dtype
        )
        cayley_t = tf.linalg.inv(eye - t * w / 2.0) @ (eye + t * w / 2.0)
        return cayley_t @ x

    def retr(self, x, u):
        xtu = utils.transposem(x) @ u
        w_ = (u - x @ xtu) @ utils.transposem(x)
        w = w_ - utils.transposem(w_)
        y = x + u
        for _ in range(self.num_iter):
            y = x + w @ ((x + y) / 2.0)
        return y

    def transp(self, x, y, v):
        return self.proju(y, v)

    def dist(self, x, y, keepdims=False):
        raise NotImplementedError

    def log(self, x, y):
        return NotImplementedError
