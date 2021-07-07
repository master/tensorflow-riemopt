import tensorflow as tf

from tensorflow_riemopt import variable
from tensorflow_riemopt.manifolds import Euclidean, Grassmannian


class VariableTest(tf.test.TestCase):
    def test_variable(self):
        euclidean = Euclidean()
        grassmannian = Grassmannian()
        with self.cached_session(use_gpu=True):
            var_1x2 = tf.Variable([[5, 3]])
            var_2x1 = tf.Variable([[5], [3]])
            self.assertEqual(
                type(variable.get_manifold(var_1x2)), type(euclidean)
            )
            self.assertEqual(
                type(variable.get_manifold(var_1x2)),
                type(variable.get_manifold(var_2x1)),
            )
            with self.assertRaises(ValueError):
                variable.assign_to_manifold(var_1x2, grassmannian)
            variable.assign_to_manifold(var_2x1, grassmannian)
            self.assertEqual(
                type(variable.get_manifold(var_2x1)), type(grassmannian)
            )
