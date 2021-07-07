import tensorflow as tf

from absl.testing import parameterized
from tensorflow.python.keras import combinations

from tensorflow_riemopt.manifolds.test_invariants import (
    TestInvariants,
    random_constant,
)
from tensorflow_riemopt.manifolds.poincare import Poincare


@combinations.generate(
    combinations.combine(
        mode=["graph", "eager"],
        manifold=[Poincare(), Poincare(k=5.0)],
        shape=[(5,), (2, 5)],
        dtype=[tf.float32, tf.float64],
    )
)
class PoincareTest(tf.test.TestCase, parameterized.TestCase):
    test_random = TestInvariants.check_random

    test_dist = TestInvariants.check_dist

    test_inner = TestInvariants.check_inner

    test_proj = TestInvariants.check_proj

    test_exp_log_inverse = TestInvariants.check_exp_log_inverse

    test_transp_retr = TestInvariants.check_transp_retr

    test_ptransp_inverse = TestInvariants.check_ptransp_inverse

    test_ptransp_inner = TestInvariants.check_ptransp_inner

    def test_oper0(self, manifold, shape, dtype):
        with self.cached_session(use_gpu=True):
            x = random_constant(shape=shape, dtype=dtype)
            u = manifold.proju(x, random_constant(shape=shape, dtype=dtype))
            z = manifold.projx(tf.zeros_like(x))
            u_at_0 = manifold.ptransp(x, z, u)
            u_ = manifold.ptransp0(x, u_at_0)
            self.assertAllCloseAccordingToType(u, u_)
            y = manifold.exp(z, u_at_0)
            y_ = manifold.exp0(u_at_0)
            self.assertAllCloseAccordingToType(y, y_)
            v = manifold.log0(y_)
            self.assertAllCloseAccordingToType(v, u_at_0)
