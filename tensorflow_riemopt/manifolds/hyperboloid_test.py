import tensorflow as tf

from absl.testing import parameterized
from tensorflow.python.keras import combinations

from tensorflow_riemopt.manifolds.test_invariants import (
    TestInvariants,
    random_constant,
)
from tensorflow_riemopt.manifolds.hyperboloid import Hyperboloid


@combinations.generate(
    combinations.combine(
        mode=["graph", "eager"],
        manifold=[Hyperboloid(), Hyperboloid(k=5.0)],
        shape=[(5,), (2, 2)],
        dtype=[tf.float64],
    )
)
class HyperboloidTest(tf.test.TestCase, parameterized.TestCase):
    test_random = TestInvariants.check_random

    test_dist = TestInvariants.check_dist

    test_inner = TestInvariants.check_inner

    test_proj = TestInvariants.check_proj

    test_exp_log_inverse = TestInvariants.check_exp_log_inverse

    test_transp_retr = TestInvariants.check_transp_retr

    test_ptransp_inverse = TestInvariants.check_ptransp_inverse

    test_ptransp_inner = TestInvariants.check_ptransp_inner

    def test_poincare(self, manifold, shape, dtype):
        with self.cached_session(use_gpu=True):
            x = manifold.projx(random_constant(shape=shape, dtype=dtype))
            y = manifold.to_poincare(x, manifold.k)
            x_ = manifold.from_poincare(y, manifold.k)
            if not tf.executing_eagerly():
                x_ = self.evaluate(x_)
            self.assertAllCloseAccordingToType(x, x_)
