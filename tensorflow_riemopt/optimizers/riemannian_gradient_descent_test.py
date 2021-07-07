"""Tests for SGD."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import combinations
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.keras import optimizers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

from tensorflow_riemopt.optimizers.riemannian_gradient_descent import (
    RiemannianSGD,
)


class RiemannianSGDOptimizerTest(test.TestCase, parameterized.TestCase):
    def testSparse(self):
        for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
            for momentum in [0.0, 0.9]:
                for nesterov in [False, True]:
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
                        opt = RiemannianSGD()
                        update = opt.apply_gradients(
                            zip([grads0, grads1], [var0, var1])
                        )
                        opt_ref = gradient_descent.SGD()
                        update_ref = opt_ref.apply_gradients(
                            zip([grads0, grads1], [var0_ref, var1_ref])
                        )
                        self.evaluate(variables.global_variables_initializer())

                        # Run 3 steps
                        for t in range(3):
                            update.run()
                            update_ref.run()

                            # Validate updated params
                            self.assertAllCloseAccordingToType(
                                self.evaluate(var0_ref), self.evaluate(var0)
                            )
                            self.assertAllCloseAccordingToType(
                                self.evaluate(var1_ref), self.evaluate(var1)
                            )

    @combinations.generate(combinations.combine(mode=["graph", "eager"]))
    def testBasic(self):
        for i, dtype in enumerate(
            [dtypes.half, dtypes.float32, dtypes.float64]
        ):
            for momentum in [0.0, 0.9]:
                for nesterov in [False, True]:
                    with self.cached_session(use_gpu=True):
                        var0_np = np.array(
                            [1.0, 2.0], dtype=dtype.as_numpy_dtype
                        )
                        grads0_np = np.array(
                            [0.1, 0.1], dtype=dtype.as_numpy_dtype
                        )
                        var1_np = np.array(
                            [3.0, 4.0], dtype=dtype.as_numpy_dtype
                        )
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
                        opt = RiemannianSGD(
                            learning_rate=learning_rate,
                            momentum=momentum,
                            nesterov=nesterov,
                        )
                        opt_ref = gradient_descent.SGD(
                            learning_rate=learning_rate,
                            momentum=momentum,
                            nesterov=nesterov,
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


if __name__ == "__main__":
    test.main()
