"""Tests for Adam."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import combinations
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.keras import optimizers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

from tensorflow_riemopt.optimizers.riemannian_adam import RiemannianAdam


def get_beta_accumulators(opt, dtype):
    local_step = math_ops.cast(opt.iterations + 1, dtype)
    beta_1_t = math_ops.cast(opt._get_hyper("beta_1"), dtype)
    beta_1_power = math_ops.pow(beta_1_t, local_step)
    beta_2_t = math_ops.cast(opt._get_hyper("beta_2"), dtype)
    beta_2_power = math_ops.pow(beta_2_t, local_step)
    return (beta_1_power, beta_2_power)


class RiemannianAdamOptimizerTest(test.TestCase, parameterized.TestCase):
    @combinations.generate(combinations.combine(mode=["graph", "eager"]))
    def testBasic(self):
        for i, dtype in enumerate(
            [dtypes.half, dtypes.float32, dtypes.float64]
        ):
            for amsgrad in [False, True]:
                with self.cached_session(use_gpu=True):
                    var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
                    grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
                    var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
                    grads1_np = np.array(
                        [0.01, 0.01], dtype=dtype.as_numpy_dtype
                    )

                    var0 = variables.Variable(var0_np, name="var0_%d" % i)
                    var1 = variables.Variable(var1_np, name="var1_%d" % i)
                    var0_ref = variables.Variable(
                        var0_np, name="var0_ref_%d" % i
                    )
                    var1_ref = variables.Variable(
                        var1_np, name="var1_ref_%d" % i
                    )
                    grads0 = constant_op.constant(grads0_np)
                    grads1 = constant_op.constant(grads1_np)

                    learning_rate = 0.001
                    beta1 = 0.9
                    beta2 = 0.999
                    epsilon = 1e-8

                    opt = RiemannianAdam(
                        learning_rate=learning_rate,
                        beta_1=beta1,
                        beta_2=beta2,
                        epsilon=epsilon,
                        amsgrad=amsgrad,
                    )
                    opt_ref = adam.Adam(
                        learning_rate=learning_rate,
                        beta_1=beta1,
                        beta_2=beta2,
                        epsilon=epsilon,
                        amsgrad=amsgrad,
                    )

                    if not context.executing_eagerly():
                        update = opt.apply_gradients(
                            zip([grads0, grads1], [var0, var1])
                        )
                        update_ref = opt_ref.apply_gradients(
                            zip([grads0, grads1], [var0_ref, var1_ref])
                        )

                    self.evaluate(variables.global_variables_initializer())
                    # Run 3 steps
                    for t in range(3):
                        beta_1_power, beta_2_power = get_beta_accumulators(
                            opt, dtype
                        )
                        self.assertAllCloseAccordingToType(
                            beta1 ** (t + 1), self.evaluate(beta_1_power)
                        )
                        self.assertAllCloseAccordingToType(
                            beta2 ** (t + 1), self.evaluate(beta_2_power)
                        )
                        if not context.executing_eagerly():
                            self.evaluate(update)
                            self.evaluate(update_ref)
                        else:
                            opt.apply_gradients(
                                zip([grads0, grads1], [var0, var1])
                            )
                            opt_ref.apply_gradients(
                                zip([grads0, grads1], [var0_ref, var1_ref])
                            )

                        # Validate updated params
                        self.assertAllCloseAccordingToType(
                            self.evaluate(var0_ref),
                            self.evaluate(var0),
                            rtol=1e-4,
                            atol=1e-4,
                        )
                        self.assertAllCloseAccordingToType(
                            self.evaluate(var1_ref),
                            self.evaluate(var1),
                            rtol=1e-4,
                            atol=1e-4,
                        )

    def testSparse(self):
        for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
            for amsgrad in [False, True]:
                with ops.Graph().as_default(), self.cached_session(
                    use_gpu=True
                ):
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

                    var0 = variables.Variable(var0_np)
                    var1 = variables.Variable(var1_np)
                    var0_ref = variables.Variable(var0_np)
                    var1_ref = variables.Variable(var1_np)
                    grads0_np_indices = np.array([0, 2], dtype=np.int32)
                    grads0 = ops.IndexedSlices(
                        constant_op.constant(grads0_np[grads0_np_indices]),
                        constant_op.constant(grads0_np_indices),
                        constant_op.constant([3]),
                    )
                    grads1_np_indices = np.array([0, 2], dtype=np.int32)
                    grads1 = ops.IndexedSlices(
                        constant_op.constant(grads1_np[grads1_np_indices]),
                        constant_op.constant(grads1_np_indices),
                        constant_op.constant([3]),
                    )
                    opt = RiemannianAdam(amsgrad=amsgrad)
                    update = opt.apply_gradients(
                        zip([grads0, grads1], [var0, var1])
                    )
                    opt_ref = adam.Adam(amsgrad=amsgrad)
                    update_ref = opt_ref.apply_gradients(
                        zip([grads0, grads1], [var0_ref, var1_ref])
                    )

                    self.evaluate(variables.global_variables_initializer())
                    beta_1_power, beta_2_power = get_beta_accumulators(
                        opt, dtype
                    )
                    # Run 3 steps
                    for t in range(3):
                        self.assertAllCloseAccordingToType(
                            0.9 ** (t + 1), self.evaluate(beta_1_power)
                        )
                        self.assertAllCloseAccordingToType(
                            0.999 ** (t + 1), self.evaluate(beta_2_power)
                        )
                        update.run()
                        update_ref.run()

                        # Validate updated params
                        self.assertAllCloseAccordingToType(
                            self.evaluate(var0_ref), self.evaluate(var0)
                        )
                        self.assertAllCloseAccordingToType(
                            self.evaluate(var1_ref), self.evaluate(var1)
                        )

    def testSharing(self):
        for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
            with ops.Graph().as_default(), self.cached_session(use_gpu=True):
                var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
                grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
                var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
                grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

                var0 = variables.Variable(var0_np)
                var1 = variables.Variable(var1_np)
                var0_ref = variables.Variable(var0_np)
                var1_ref = variables.Variable(var1_np)
                grads0 = constant_op.constant(grads0_np)
                grads1 = constant_op.constant(grads1_np)
                opt = RiemannianAdam()
                update1 = opt.apply_gradients(
                    zip([grads0, grads1], [var0, var1])
                )
                update2 = opt.apply_gradients(
                    zip([grads0, grads1], [var0, var1])
                )

                update1_ref = adam.Adam().apply_gradients(
                    zip([grads0], [var0_ref])
                )
                update2_ref = adam.Adam().apply_gradients(
                    zip([grads1], [var1_ref])
                )

                self.evaluate(variables.global_variables_initializer())
                beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
                # Run 3 steps
                for t in range(3):
                    self.assertAllCloseAccordingToType(
                        0.9 ** (t + 1), self.evaluate(beta_1_power)
                    )
                    self.assertAllCloseAccordingToType(
                        0.999 ** (t + 1), self.evaluate(beta_2_power)
                    )
                    if t % 2 == 0:
                        update1.run()
                    else:
                        update2.run()

                    update1_ref.run()
                    update2_ref.run()

                    # Validate updated params
                    self.assertAllCloseAccordingToType(
                        self.evaluate(var0_ref), self.evaluate(var0)
                    )
                    self.assertAllCloseAccordingToType(
                        self.evaluate(var1_ref), self.evaluate(var1)
                    )


if __name__ == "__main__":
    test.main()
