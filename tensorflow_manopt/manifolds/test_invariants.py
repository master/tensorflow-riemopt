"""Manifold test invariants."""
import tensorflow as tf
import numpy as np


def random_constant(shape, dtype):
    return tf.constant(
        np.random.uniform(size=shape, high=1e-1),
        dtype=dtype.as_numpy_dtype,
    )


class TestInvariants(tf.test.TestCase):
    def check_random(self, manifold, shape, dtype):
        """Check random point generator"""
        with self.cached_session(use_gpu=True):
            x = manifold.random(shape=shape, dtype=dtype)
            self.assertEqual(list(shape), x.shape.as_list())
            x_on_manifold = manifold.check_point_on_manifold(x)
            if not tf.executing_eagerly():
                x_on_manifold = self.evaluate(x_on_manifold)
                self.assertTrue(x_on_manifold)

    def check_dist(self, manifold, shape, dtype):
        """Check the distance axioms"""
        with self.cached_session(use_gpu=True):
            x_rand = random_constant(shape=shape, dtype=dtype)
            x = manifold.projx(x_rand)
            y_rand = random_constant(shape=shape, dtype=dtype)
            y = manifold.projx(x_rand)
            dist_xy = manifold.dist(x, y)
            dist_yx = manifold.dist(y, x)
            self.assertAllCloseAccordingToType(dist_xy, dist_yx)
            dist_xx = manifold.dist(x, x)
            # low precision comparison for manifolds which use trigonometric functions
            self.assertAllClose(dist_xx, tf.zeros_like(dist_xx), atol=1e-3)
            orig_shape = x.shape.as_list()
            keepdims_shape = manifold.dist(x, x, keepdims=True).shape.as_list()
            nokeepdims_shape = manifold.dist(
                x, x, keepdims=False
            ).shape.as_list()
            self.assertEqual(len(keepdims_shape), len(orig_shape))
            self.assertEqual(
                len(nokeepdims_shape), len(orig_shape) - manifold.ndims
            )
            if manifold.ndims > 0:
                self.assertEqual(
                    keepdims_shape[-manifold.ndims :], [1] * manifold.ndims
                )

    def check_inner(self, manifold, shape, dtype):
        """Check the inner product axioms"""
        with self.cached_session(use_gpu=True):
            x_rand = random_constant(shape=shape, dtype=dtype)
            x = manifold.projx(x_rand)
            u = manifold.proju(x, x_rand)
            orig_shape = u.shape.as_list()
            keepdims_shape = manifold.inner(
                x, u, u, keepdims=True
            ).shape.as_list()
            nokeepdims_shape = manifold.inner(
                x, u, u, keepdims=False
            ).shape.as_list()
            self.assertEqual(len(keepdims_shape), len(orig_shape))
            self.assertEqual(
                len(nokeepdims_shape), len(orig_shape) - manifold.ndims
            )
            if manifold.ndims > 0:
                self.assertEqual(
                    keepdims_shape[-manifold.ndims :], [1] * manifold.ndims
                )

    def check_proj(self, manifold, shape, dtype):
        """Check projection from the ambient space"""
        with self.cached_session(use_gpu=True):
            x_rand = random_constant(shape=shape, dtype=dtype)
            x = manifold.projx(x_rand)
            x_on_manifold = manifold.check_point_on_manifold(x)
            if not tf.executing_eagerly():
                x_on_manifold = self.evaluate(x_on_manifold)
            self.assertTrue(x_on_manifold)
            u_rand = random_constant(shape=shape, dtype=dtype)
            u = manifold.proju(x, u_rand)
            u_on_tangent = manifold.check_vector_on_tangent(x, u)
            if not tf.executing_eagerly():
                u_on_tangent = self.evaluate(u_on_tangent)
            self.assertTrue(u_on_tangent)

    def check_exp_log_inverse(self, manifold, shape, dtype):
        """Check that logarithmic map is the inverse of exponential map"""
        with self.cached_session(use_gpu=True):
            x = manifold.projx(random_constant(shape, dtype))
            u = manifold.proju(x, random_constant(shape, dtype))
            y = manifold.exp(x, u)
            y_on_manifold = manifold.check_point_on_manifold(y)
            if not tf.executing_eagerly():
                y_on_manifold = self.evaluate(y_on_manifold)
            self.assertTrue(y_on_manifold)
            v = manifold.log(x, y)
            v_on_tangent = manifold.check_vector_on_tangent(x, v)
            if not tf.executing_eagerly():
                v_on_tangent = self.evaluate(v_on_tangent)
            self.assertTrue(v_on_tangent)
            self.assertAllCloseAccordingToType(u, v)

    def check_transp_retr(self, manifold, shape, dtype):
        """Test that vector transport is compatible with retraction"""
        with self.cached_session(use_gpu=True):
            x = manifold.projx(random_constant(shape, dtype))
            u = manifold.proju(x, random_constant(shape, dtype))
            y = manifold.retr(x, u)
            y_on_manifold = manifold.check_point_on_manifold(y)
            if not tf.executing_eagerly():
                y_on_manifold = self.evaluate(y_on_manifold)
            v = manifold.proju(x, random_constant(shape, dtype))
            v_ = manifold.transp(x, y, v)
            v_on_tangent = manifold.check_vector_on_tangent(y, v_)
            if not tf.executing_eagerly():
                v_on_tangent = self.evaluate(v_on_tangent)
            self.assertTrue(v_on_tangent)
            w = manifold.proju(x, random_constant(shape, dtype))
            w_ = manifold.transp(x, y, w)
            w_v = v + w
            w_v_ = manifold.transp(x, y, w_v)
            self.assertAllCloseAccordingToType(w_v_, w_ + v_)

    def check_ptransp_inverse(self, manifold, shape, dtype):
        """Test that parallel transport is an invertible operation"""
        with self.cached_session(use_gpu=True):
            x = manifold.projx(random_constant(shape, dtype))
            y = manifold.projx(random_constant(shape, dtype))
            u = manifold.proju(x, random_constant(shape, dtype))
            v = manifold.ptransp(x, y, u)
            v_on_tangent = manifold.check_vector_on_tangent(y, v)
            if not tf.executing_eagerly():
                v_on_tangent = self.evaluate(v_on_tangent)
            self.assertTrue(v_on_tangent)
            w = manifold.ptransp(y, x, v)
            w_on_tangent = manifold.check_vector_on_tangent(x, w)
            if not tf.executing_eagerly():
                w_on_tangent = self.evaluate(w_on_tangent)
            self.assertTrue(w_on_tangent)
            self.assertAllCloseAccordingToType(u, w)

    def check_ptransp_inner(self, manifold, shape, dtype):
        """Check that parallel transport preserves the inner product"""
        with self.cached_session(use_gpu=True):
            x = manifold.projx(random_constant(shape, dtype))
            y = manifold.projx(random_constant(shape, dtype))
            u = manifold.proju(x, random_constant(shape, dtype))
            v = manifold.proju(x, random_constant(shape, dtype))
            uv = manifold.inner(x, u, v)
            u_ = manifold.ptransp(x, y, u)
            v_ = manifold.ptransp(x, y, v)
            u_v_ = manifold.inner(y, u_, v_)
            self.assertAllCloseAccordingToType(uv, u_v_)

    def check_geodesic(self, manifold, shape, dtype):
        """Check that the exponential map lies on a geodesic"""
        with self.cached_session(use_gpu=True):
            x = manifold.projx(random_constant(shape=shape, dtype=dtype))
            u = manifold.proju(x, random_constant(shape=shape, dtype=dtype))
            y = manifold.geodesic(x, u, 1.0)
            y_on_manifold = manifold.check_point_on_manifold(y)
            if not tf.executing_eagerly():
                y_on_manifold = self.evaluate(y_on_manifold)
            self.assertTrue(y_on_manifold)
            y_ = manifold.exp(x, u)
            self.assertAllCloseAccordingToType(y, y_)

    def check_pairmean(self, manifold, shape, dtype):
        """Check that the Riemannian mean is equidistant from points"""
        with self.cached_session(use_gpu=True):
            x_rand = random_constant(shape=shape, dtype=dtype)
            x = manifold.projx(x_rand)
            y_rand = random_constant(shape=shape, dtype=dtype)
            y = manifold.projx(x_rand)
            m = manifold.pairmean(x, y)
            m_on_manifold = manifold.check_point_on_manifold(m)
            if not tf.executing_eagerly():
                m_on_manifold = self.evaluate(m_on_manifold)
            self.assertTrue(m_on_manifold)
            dist_x_m = manifold.dist(x, m)
            dist_y_m = manifold.dist(y, m)
            self.assertAllCloseAccordingToType(dist_x_m, dist_y_m)
