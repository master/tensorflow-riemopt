"""Tests for RMSprop."""

from absl.testing import parameterized
import keras
import numpy as np
import tensorflow as tf

from tensorflow_riemopt.manifolds import StiefelEuclidean
from tensorflow_riemopt.optimizers.constrained_rmsprop import (
    ConstrainedRMSprop,
)
from tensorflow_riemopt.variable import assign_to_manifold


class ConstrainedRMSpropOptimizerTest(tf.test.TestCase, parameterized.TestCase):
    def testEuclidean(self):
        for dtype in [tf.half, tf.float32, tf.float64]:
            for centered in [False, True]:
                for sparse in [False, True]:
                    var0_np = np.array(
                        [1.0, 1.0, 2.0], dtype=dtype.as_numpy_dtype
                    )
                    grads0_np = np.array(
                        [0.1, 0.0, 0.1], dtype=dtype.as_numpy_dtype
                    )
                    var1_np = np.array(
                        [3.0, 3.0, 4.0], dtype=dtype.as_numpy_dtype
                    )
                    grads1_np = np.array(
                        [0.01, 0.0, 0.01], dtype=dtype.as_numpy_dtype
                    )

                    var0 = tf.Variable(var0_np)
                    var1 = tf.Variable(var1_np)
                    var0_ref = tf.Variable(var0_np)
                    var1_ref = tf.Variable(var1_np)
                    grads0 = tf.constant(grads0_np)
                    grads1 = tf.constant(grads1_np)
                    if sparse:
                        indices = np.array([0, 2], dtype=np.int32)
                        grads0 = tf.IndexedSlices(
                            tf.constant(grads0_np[indices]),
                            tf.constant(indices),
                            tf.constant([3]),
                        )
                        grads1 = tf.IndexedSlices(
                            tf.constant(grads1_np[indices]),
                            tf.constant(indices),
                            tf.constant([3]),
                        )

                    learning_rate = 0.001
                    opt = ConstrainedRMSprop(
                        learning_rate=learning_rate,
                        centered=centered,
                    )
                    opt_ref = keras.optimizers.RMSprop(
                        learning_rate=learning_rate,
                        centered=centered,
                    )

                    # Run 3 steps
                    for t in range(3):
                        opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
                        opt_ref.apply_gradients(
                            zip([grads0, grads1], [var0_ref, var1_ref])
                        )

                        # Validate updated params
                        self.assertAllCloseAccordingToType(
                            var0_ref.numpy(), var0.numpy()
                        )
                        self.assertAllCloseAccordingToType(
                            var1_ref.numpy(), var1.numpy()
                        )

    def testManifold(self):
        manifold = StiefelEuclidean()
        for dtype in [tf.float32, tf.float64]:
            for centered in [False, True]:
                var0_np = manifold.random([5, 3], dtype=dtype).numpy()
                var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
                grads0 = tf.random.normal([5, 3], dtype=dtype)
                grads1 = tf.constant([0.01, 0.01], dtype=dtype)

                var0 = tf.Variable(var0_np)
                var1 = tf.Variable(var1_np)
                var1_ref = tf.Variable(var1_np)
                assign_to_manifold(var0, manifold)

                opt = ConstrainedRMSprop(learning_rate=0.001, centered=centered)
                opt_ref = keras.optimizers.RMSprop(
                    learning_rate=0.001, centered=centered
                )

                # Run 3 steps
                for t in range(3):
                    opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
                    opt_ref.apply_gradients([(grads1, var1_ref)])

                    # Validate updated params
                    self.assertTrue(np.all(np.isfinite(var0.numpy())))
                    self.assertTrue(
                        manifold.check_point_on_manifold(
                            var0, atol=1e-5, rtol=1e-5
                        )
                    )
                    self.assertNotAllClose(var0_np, var0.numpy())
                    self.assertAllCloseAccordingToType(
                        var1_ref.numpy(), var1.numpy()
                    )

    def testCompileFit(self):
        manifold = StiefelEuclidean()

        class Layer(keras.layers.Layer):
            def build(self, input_shape):
                self.w = self.add_weight(
                    name="w", shape=[4, 2], initializer=manifold.random
                )
                assign_to_manifold(self.w, manifold)

            def call(self, inputs):
                return inputs @ self.w

        layer = Layer()
        model = keras.Sequential(
            [keras.Input((4,)), layer, keras.layers.Dense(1)]
        )
        model.compile(
            optimizer=ConstrainedRMSprop(0.1, stabilize=2), loss="mse"
        )
        model.fit(
            tf.random.normal([8, 4]),
            tf.random.normal([8, 1]),
            epochs=1,
            batch_size=4,
            verbose=0,
        )
        self.assertTrue(
            manifold.check_point_on_manifold(
                layer.w.value, atol=1e-5, rtol=1e-5
            )
        )

    def testConfig(self):
        opt = ConstrainedRMSprop(
            learning_rate=0.1,
            rho=0.8,
            epsilon=1e-8,
            centered=True,
            stabilize=2,
        )
        opt_new = keras.optimizers.deserialize(keras.optimizers.serialize(opt))
        self.assertIsInstance(opt_new, ConstrainedRMSprop)
        self.assertAllClose(0.1, opt_new.learning_rate)
        self.assertEqual(0.8, opt_new.rho)
        self.assertEqual(1e-8, opt_new.epsilon)
        self.assertEqual(True, opt_new.centered)
        self.assertEqual(2, opt_new.stabilize)


if __name__ == "__main__":
    tf.test.main()
