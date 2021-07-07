import tensorflow as tf

from absl.testing import parameterized
from tensorflow.python.keras import combinations

from tensorflow_riemopt.manifolds.test_invariants import TestInvariants
from tensorflow_riemopt.manifolds.sphere import Sphere
from tensorflow_riemopt.manifolds.special_orthogonal import SpecialOrthogonal
from tensorflow_riemopt.manifolds.approximate_mixin import ApproximateMixin


class SpherePoleLadder(ApproximateMixin, Sphere):
    def transp(self, x, y, v):
        return self.ladder_transp(x, y, v, method="pole", n_steps=1)


class SphereSchildLadder(ApproximateMixin, Sphere):
    def transp(self, x, y, v):
        return self.ladder_transp(x, y, v, method="schild", n_steps=1)


@combinations.generate(
    combinations.combine(
        mode=["graph", "eager"],
        manifold=[SpherePoleLadder()],
        shape=[(3,), (3, 3)],
        dtype=[tf.float32, tf.float64],
    )
)
class PoleLadderTest(tf.test.TestCase, parameterized.TestCase):
    test_ptransp_inverse = TestInvariants.check_ptransp_inverse

    test_ptransp_inner = TestInvariants.check_ptransp_inner


@combinations.generate(
    combinations.combine(
        mode=["graph", "eager"],
        manifold=[SphereSchildLadder()],
        shape=[(3,), (3, 3)],
        dtype=[tf.float32, tf.float64],
    )
)
class SchildLadderTest(tf.test.TestCase, parameterized.TestCase):
    test_ptransp_inverse = TestInvariants.check_ptransp_inverse
