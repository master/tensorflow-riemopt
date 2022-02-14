"""Manifolds of symmetric positive definite matrices."""
import warnings

import tensorflow as tf

from tensorflow_riemopt.manifolds.manifold import Manifold
from tensorflow_riemopt.manifolds.cholesky import Cholesky
from tensorflow_riemopt.manifolds import utils


class _SymmetricPositiveDefinite(Manifold):
    """Manifold of real symmetric positive definite matrices.

    Henrion, Didier, and Jérôme Malick. "Projection methods in conic
    optimization." Handbook on Semidefinite, Conic and Polynomial
    Optimization. Springer, Boston, MA, 2012. 565-600.

    """

    ndims = 2

    def _check_shape(self, shape):
        return shape[-1] == shape[-2]

    def _check_point_on_manifold(self, x, atol, rtol):
        x_t = utils.transposem(x)
        eigvals, _ = tf.linalg.eigh(x)
        is_symmetric = utils.allclose(x, x_t, atol, rtol)
        is_pos_vals = utils.allclose(eigvals, tf.abs(eigvals), atol, rtol)
        is_zero_vals = utils.allclose(
            eigvals, tf.zeros_like(eigvals), atol, rtol
        )
        return is_symmetric & is_pos_vals & tf.logical_not(is_zero_vals)

    def _check_vector_on_tangent(self, x, u, atol, rtol):
        u_t = utils.transposem(u)
        return utils.allclose(u, u_t, atol, rtol)

    def projx(self, x):
        warnings.warn(
            (
                "{}.projx performs a projection onto the open set of"
                + " PSD matrices"
            ).format(self.__class__.__name__)
        )
        x_sym = (utils.transposem(x) + x) / 2.0
        s, u, v = tf.linalg.svd(x_sym)
        sigma = tf.linalg.diag(tf.maximum(s, 0.0))
        return v @ sigma @ utils.transposem(v)

    def proju(self, x, u):
        return 0.5 * (utils.transposem(u) + u)


class SPDAffineInvariant(_SymmetricPositiveDefinite):
    """Manifold of symmetric positive definite matrices endowed with the
    affine-invariant metric.

    Pennec, Xavier, Pierre Fillard, and Nicholas Ayache. "A Riemannian
    framework for tensor computing." International Journal of computer vision
    66.1 (2006): 41-66.

    Sra, Suvrit, and Reshad Hosseini. "Conic geometric optimization on the
    manifold of positive definite matrices." SIAM Journal on Optimization 25.1
    (2015): 713-739.

    """

    name = "Affine-Invariant SPD"

    def dist(self, x, y, keepdims=False):
        x_sqrt_inv = tf.linalg.inv(tf.linalg.sqrtm(x))
        log = utils.logm(x_sqrt_inv @ y @ x_sqrt_inv)
        return tf.linalg.norm(log, axis=[-2, -1], ord="fro", keepdims=keepdims)

    def inner(self, x, u, v, keepdims=False):
        x_sqrt_inv = tf.linalg.inv(tf.linalg.sqrtm(x))
        x_inv = tf.linalg.inv(x)
        traces = tf.linalg.trace(x_sqrt_inv @ u @ x_inv @ v @ x_sqrt_inv)
        return traces[..., tf.newaxis, tf.newaxis] if keepdims else traces

    def exp(self, x, u):
        x_sqrt = tf.linalg.sqrtm(x)
        x_sqrt_inv = tf.linalg.inv(x_sqrt)
        return x_sqrt @ tf.linalg.expm(x_sqrt_inv @ u @ x_sqrt_inv) @ x_sqrt

    retr = exp

    def log(self, x, y):
        x_sqrt = tf.linalg.sqrtm(x)
        x_sqrt_inv = tf.linalg.inv(x_sqrt)
        return x_sqrt @ utils.logm(x_sqrt_inv @ y @ x_sqrt_inv) @ x_sqrt

    def geodesic(self, x, u, t):
        x_sqrt = tf.linalg.sqrtm(x)
        x_sqrt_inv = tf.linalg.inv(x_sqrt)
        return x_sqrt @ tf.linalg.expm(t * x_sqrt_inv @ u @ x_sqrt_inv) @ x_sqrt

    def ptransp(self, x, y, v):
        e = tf.linalg.sqrtm(y @ tf.linalg.inv(x))
        return e @ v @ utils.transposem(e)

    def ptransp1(self, x, y, v):
        x_sqrt = tf.linalg.sqrtm(x)
        x_sqrt_inv = tf.linalg.inv(x_sqrt)
        h = tf.linalg.expm(0.5 * x_sqrt_inv @ self.log(x, y) @ x_sqrt_inv)
        return x_sqrt @ h @ x_sqrt_inv @ v @ x_sqrt_inv @ h @ x_sqrt

    transp = ptransp

    def pairmean(self, x, y):
        x_sqrt = tf.linalg.sqrtm(x)
        x_sqrt_inv = tf.linalg.inv(x_sqrt)
        return x_sqrt @ tf.linalg.sqrtm(x_sqrt_inv @ y @ x_sqrt_inv) @ x_sqrt


class SPDLogEuclidean(_SymmetricPositiveDefinite):
    """Manifold of symmetric positive definite matrices endowed with the
    Log-Euclidean metric.

    Arsigny, Vincent, et al. "Geometric means in a novel vector space
    structure on symmetric positive-definite matrices." SIAM journal on matrix
    analysis and applications 29.1 (2007): 328-347.

    """

    def dist(self, x, y, keepdims=False):
        diff = utils.logm(y) - utils.logm(x)
        return tf.linalg.norm(diff, axis=[-2, -1], ord="fro", keepdims=keepdims)

    def _diff_power(self, x, d, power):
        e, v = tf.linalg.eigh(x)
        v_t = utils.transposem(v)
        e = tf.expand_dims(e, -2)
        if power == "log":
            pow_e = tf.math.log(e)
        elif power == "exp":
            pow_e = tf.math.exp(e)
        s = utils.transposem(tf.ones_like(e)) @ e
        pow_s = utils.transposem(tf.ones_like(pow_e)) @ pow_e
        denom = utils.transposem(s) - s
        numer = utils.transposem(pow_s) - pow_s
        abs_denom = tf.math.abs(denom)
        eps = utils.get_eps(x)
        if power == "log":
            numer = tf.where(abs_denom < eps, tf.ones_like(numer), numer)
            denom = tf.where(abs_denom < eps, utils.transposem(s), denom)
        elif power == "exp":
            numer = tf.where(abs_denom < eps, utils.transposem(pow_s), numer)
            denom = tf.where(abs_denom < eps, tf.ones_like(denom), denom)
        t = v_t @ d @ v * numer / denom
        return v @ t @ v_t

    def _diff_exp(self, x, d):
        """Directional derivative of `expm` at :math:`x` along :math:`d`"""
        return self._diff_power(x, d, "exp")

    def _diff_log(self, x, d):
        """Directional derivative of `logm` at :math:`x` along :math:`d`"""
        return self._diff_power(x, d, "log")

    def inner(self, x, u, v, keepdims=False):
        dlog_u = self._diff_log(x, u)
        dlog_v = self._diff_log(x, v)
        return tf.reduce_sum(dlog_u * dlog_v, axis=[-2, -1], keepdims=keepdims)

    def exp(self, x, u):
        return tf.linalg.expm(utils.logm(x) + self._diff_log(x, u))

    retr = exp

    def log(self, x, y):
        return self._diff_exp(utils.logm(x), utils.logm(y) - utils.logm(x))

    def geodesic(self, x, u, t):
        return tf.linalg.expm(utils.logm(x) + t * self._diff_log(x, u))

    def pairmean(self, x, y):
        return tf.linalg.expm((utils.logm(x) + utils.logm(y)) / 2.0)

    def transp(self, x, y, v):
        raise NotImplementedError


class SPDLogCholesky(_SymmetricPositiveDefinite):
    """Manifold of symmetric positive definite matrices endowed with the
    Log-Cholesky metric.

    The geometry of manifold is induced by the diffeomorphism between the SPD
    and Cholesky spaces.

    Lin, Zhenhua. "Riemannian Geometry of Symmetric Positive Definite Matrices
    via Cholesky Decomposition." SIAM Journal on Matrix Analysis and
    Applications 40.4 (2019): 1353-1370.

    """

    name = "Log-Cholesky SPD"

    def __init__(self):
        self._cholesky = Cholesky()
        super().__init__()

    def to_cholesky(self, x):
        """Diffeomorphism that maps to the Cholesky space"""
        assert_x = tf.debugging.Assert(self.check_point_on_manifold(x), [x])
        with tf.control_dependencies([assert_x]):
            return tf.linalg.cholesky(x)

    def from_cholesky(self, x):
        """Inverse of the diffeomorphism to the Cholesky space"""
        assert_x = tf.debugging.Assert(
            self._cholesky.check_point_on_manifold(x), [x]
        )
        with tf.control_dependencies([assert_x]):
            return x @ utils.transposem(x)

    def diff_to_cholesky(self, x, u):
        """Differential of the diffeomorphism to the Cholesky space"""
        assert_x = tf.debugging.Assert(self.check_point_on_manifold(x), [x])
        assert_u = tf.debugging.Assert(self.check_vector_on_tangent(x, u), [u])
        with tf.control_dependencies([assert_x, assert_u]):
            y = self.to_cholesky(x)
            y_inv = tf.linalg.inv(y)
            p = y_inv @ u @ utils.transposem(y_inv)
            p_diag, p_lower = self._cholesky._diag_and_strictly_lower(p)
            return y @ (p_lower + 0.5 * tf.linalg.diag(p_diag))

    def diff_from_cholesky(self, x, u):
        """Inverse of the differential of diffeomorphism to the Cholesky space"""
        assert_x = tf.debugging.Assert(
            self._cholesky.check_point_on_manifold(x), [x]
        )
        assert_u = tf.debugging.Assert(
            self._cholesky.check_vector_on_tangent(x, u), [u]
        )
        with tf.control_dependencies([assert_x, assert_u]):
            return x @ utils.transposem(u) + u @ utils.transposem(x)

    def dist(self, x, y, keepdims=False):
        x_chol = self.to_cholesky(x)
        y_chol = self.to_cholesky(y)
        return self._cholesky.dist(x_chol, y_chol, keepdims=keepdims)

    def inner(self, x, u, v, keepdims=False):
        x_chol = self.to_cholesky(x)
        u_diff_chol = self.diff_to_cholesky(x, u)
        v_diff_chol = self.diff_to_cholesky(x, v)
        return self._cholesky.inner(
            x_chol, u_diff_chol, v_diff_chol, keepdims=keepdims
        )

    def exp(self, x, u):
        x_chol = self.to_cholesky(x)
        u_diff_chol = self.diff_to_cholesky(x, u)
        exp_chol = self._cholesky.exp(x_chol, u_diff_chol)
        return self.from_cholesky(exp_chol)

    retr = exp

    def log(self, x, y):
        x_chol = self.to_cholesky(x)
        y_chol = self.to_cholesky(y)
        log_chol = self._cholesky.log(x_chol, y_chol)
        return self.diff_from_cholesky(x_chol, log_chol)

    def ptransp(self, x, y, v):
        x_chol = self.to_cholesky(x)
        y_chol = self.to_cholesky(y)
        v_diff_chol = self.diff_to_cholesky(x, v)
        transp_chol = self._cholesky.transp(x_chol, y_chol, v_diff_chol)
        return self.diff_from_cholesky(y_chol, transp_chol)

    transp = ptransp

    def geodesic(self, x, u, t):
        x_chol = self.to_cholesky(x)
        u_diff_chol = self.diff_to_cholesky(x, u)
        geodesic_chol = self._cholesky.geodesic(x_chol, u_diff_chol, t)
        return self.from_cholesky(geodesic_chol)

    def pairmean(self, x, y):
        x_chol = self.to_cholesky(x)
        y_chol = self.to_cholesky(y)
        pairmean_chol = self._cholesky.pairmean(x_chol, y_chol)
        return self.from_cholesky(pairmean_chol)
