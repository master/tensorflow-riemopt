import tensorflow as tf

from absl.testing import parameterized
from tensorflow.python.keras import combinations

from tensorflow_riemopt.manifolds.test_invariants import TestInvariants
from tensorflow_riemopt.manifolds.stiefel import StiefelEuclidean
from tensorflow_riemopt.manifolds.stiefel import StiefelCanonical
from tensorflow_riemopt.manifolds.stiefel import StiefelCayley


@combinations.generate(
    combinations.combine(
        mode=["graph", "eager"],
        manifold=[StiefelEuclidean()],
        shape=[(2, 5, 3), (5, 3)],
        dtype=[tf.float32, tf.float64],
    )
)
class StiefelEuclideanTest(tf.test.TestCase, parameterized.TestCase):
    test_random = TestInvariants.check_random

    test_config = TestInvariants.check_config

    test_inner = TestInvariants.check_inner

    test_proj = TestInvariants.check_proj

    test_transp_retr = TestInvariants.check_transp_retr


@combinations.generate(
    combinations.combine(
        mode=["graph", "eager"],
        manifold=[StiefelCanonical(num_iter=20)],
        shape=[(2, 5, 3), (5, 3)],
        dtype=[tf.float32, tf.float64],
    )
)
class StiefelCanonicalTest(tf.test.TestCase, parameterized.TestCase):
    test_random = TestInvariants.check_random

    test_config = TestInvariants.check_config

    test_dist = TestInvariants.check_dist

    test_inner = TestInvariants.check_inner

    test_proj = TestInvariants.check_proj

    test_exp_log_inverse = TestInvariants.check_exp_log_inverse

    test_transp_retr = TestInvariants.check_transp_retr

    test_geodesic = TestInvariants.check_geodesic

    test_pairmean = TestInvariants.check_pairmean


@combinations.generate(
    combinations.combine(
        mode=["graph", "eager"],
        manifold=[StiefelCayley(num_iter=100)],
        shape=[(2, 5, 3), (5, 3)],
        dtype=[tf.float32, tf.float64],
    )
)
class StiefelCayleyTest(tf.test.TestCase, parameterized.TestCase):
    test_random = TestInvariants.check_random

    test_config = TestInvariants.check_config

    test_inner = TestInvariants.check_inner

    test_proj = TestInvariants.check_proj

    test_transp_retr = TestInvariants.check_transp_retr
