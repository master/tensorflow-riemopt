import tensorflow as tf

from absl.testing import parameterized
from tensorflow.python.keras import combinations

from tensorflow_riemopt.manifolds.test_invariants import TestInvariants
from tensorflow_riemopt.manifolds.product import Product
from tensorflow_riemopt.manifolds.sphere import Sphere
from tensorflow_riemopt.manifolds.euclidean import Euclidean
from tensorflow_riemopt.manifolds.stiefel import StiefelEuclidean


@combinations.generate(
    combinations.combine(
        mode=["graph", "eager"],
        manifold=[
            Product((Sphere(), (3,)), (Sphere(), (2,))),
            Product((Euclidean(), (3,)), (Sphere(), (2,))),
        ],
        shape=[(5,), (2, 5), (2, 2, 5)],
        dtype=[tf.float32, tf.float64],
    )
)
class ProductTest(tf.test.TestCase, parameterized.TestCase):
    test_random = TestInvariants.check_random

    test_config = TestInvariants.check_config

    test_dist = TestInvariants.check_dist

    test_inner = TestInvariants.check_inner

    test_proj = TestInvariants.check_proj

    test_egrad2rgrad = TestInvariants.check_egrad2rgrad

    test_exp_log_inverse = TestInvariants.check_exp_log_inverse

    test_transp_retr = TestInvariants.check_transp_retr

    test_ptransp_inverse = TestInvariants.check_ptransp_inverse

    test_ptransp_inner = TestInvariants.check_ptransp_inner


@combinations.generate(
    combinations.combine(
        mode=["graph", "eager"],
        manifold=[Product((StiefelEuclidean(), (5, 3)), (Sphere(), (2,)))],
        shape=[(17,), (2, 17)],
        dtype=[tf.float32, tf.float64],
    )
)
class ProductStiefelTest(tf.test.TestCase, parameterized.TestCase):
    test_random = TestInvariants.check_random

    test_config = TestInvariants.check_config

    test_inner = TestInvariants.check_inner

    test_proj = TestInvariants.check_proj

    test_egrad2rgrad = TestInvariants.check_egrad2rgrad

    test_transp_retr = TestInvariants.check_transp_retr
