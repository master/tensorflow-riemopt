"""Constrained RMSprop optimizer implementation.

Kumar Roy, Soumava, Zakaria Mhammedi, and Mehrtash Harandi. "Geometry aware
constrained optimization techniques for deep learning." Proceedings of the
IEEE Conference on Computer Vision and Pattern Recognition. 2018.
"""
import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import gen_training_ops
from keras.optimizer_v2.optimizer_v2 import OptimizerV2

from tensorflow_riemopt.variable import get_manifold


@generic_utils.register_keras_serializable(name="ConstrainedRMSprop")
class ConstrainedRMSprop(OptimizerV2):
    """Optimizer that implements the RMSprop algorithm."""

    _HAS_AGGREGATE_GRAD = True

    def __init__(
        self,
        learning_rate=0.001,
        rho=0.9,
        epsilon=1e-7,
        centered=False,
        stabilize=None,
        name="ConstrainedRMSprop",
        **kwargs,
    ):
        """Construct a new Constrained RMSprop optimizer.

        Kumar Roy, Soumava, Zakaria Mhammedi, and Mehrtash Harandi. "Geometry
        aware constrained optimization techniques for deep learning." Proceedings
        of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.

        Args:
          learning_rate: A `Tensor`, floating point value, or a schedule that is a
            `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
            that takes no arguments and returns the actual value to use. The
            learning rate. Defeaults to 0.001.
          rho: Discounting factor for the history/coming gradient. Defaults to 0.9.
          epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
            1e-7.
          centered: Boolean. If `True`, gradients are normalized by the estimated
            variance of the gradient; if False, by the uncentered second moment.
            Setting this to `True` may help with training, but is slightly more
            expensive in terms of computation and memory. Defaults to `False`.
          stabilize: Project variables back to manifold every `stabilize` steps.
            Defaults to `None`.
          name: Optional name prefix for the operations created when applying
            gradients. Defaults to "ConstrainedRMSprop".
          **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
            `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
            gradients by value, `decay` is included for backward compatibility to
            allow time inverse decay of learning rate. `lr` is included for backward
            compatibility, recommended to use `learning_rate` instead.
        """
        super(ConstrainedRMSprop, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("rho", rho)

        self.epsilon = epsilon or backend_config.epsilon()
        self.centered = centered
        self.stabilize = stabilize

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "rms")
        if self.centered:
            for var in var_list:
                self.add_slot(var, "mg")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(ConstrainedRMSprop, self)._prepare_local(
            var_device, var_dtype, apply_state
        )

        rho = array_ops.identity(self._get_hyper("rho", var_dtype))
        apply_state[(var_device, var_dtype)].update(
            dict(
                neg_lr_t=-apply_state[(var_device, var_dtype)]["lr_t"],
                epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                rho=rho,
                one_minus_rho=1.0 - rho,
            )
        )

    @def_function.function(experimental_compile=True)
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        manifold = get_manifold(var)
        grad = manifold.egrad2rgrad(var, grad)
        grad_sq = manifold.egrad2rgrad(var, grad * grad)

        rms = self.get_slot(var, "rms")
        rms_t = (
            coefficients["rho"] * rms + coefficients["one_minus_rho"] * grad_sq
        )
        denom_t = rms_t
        if self.centered:
            mg = self.get_slot(var, "mg")
            mg_t = (
                coefficients["rho"] * mg + coefficients["one_minus_rho"] * grad
            )
            denom_t -= math_ops.square(mg_t)

        var_t = manifold.retr(
            var,
            -coefficients["lr_t"]
            * grad
            / (math_ops.sqrt(denom_t) + coefficients["epsilon"]),
        )
        rms.assign(manifold.transp(var, var_t, rms_t))
        if self.centered:
            mg.assign(manifold.transp(var, var_t, mg_t))
        var.assign(var_t)

        if self.stabilize is not None:
            self._stabilize(var)

    @def_function.function(experimental_compile=True)
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        manifold = get_manifold(var)
        grad = manifold.egrad2rgrad(var, grad)
        grad_sq = manifold.egrad2rgrad(var, grad * grad)

        rms = self.get_slot(var, "rms")
        rms_scaled_g_values = grad_sq * coefficients["one_minus_rho"]
        rms_t_values = (
            array_ops.gather(rms, indices) * coefficients["rho"]
            + rms_scaled_g_values
        )

        denom_t = rms_t_values
        if self.centered:
            mg = self.get_slot(var, "mg")
            mg_scaled_g_values = grad * coefficients["one_minus_rho"]
            mg_t_values = (
                array_ops.gather(mg, indices) * coefficients["rho"]
                + mg_scaled_g_values
            )
            denom_t -= math_ops.square(mg_t_values)

        var_values = array_ops.gather(var, indices)
        var_t_values = manifold.retr(
            var_values,
            coefficients["neg_lr_t"]
            * grad
            / (math_ops.sqrt(denom_t) + coefficients["epsilon"]),
        )

        rms_t_transp = manifold.transp(var_values, var_t_values, rms_t_values)
        rms.scatter_update(ops.IndexedSlices(rms_t_transp, indices))

        if self.centered:
            mg_t_transp = manifold.transp(var_values, var_t_values, mg_t_values)
            mg.scatter_update(ops.IndexedSlices(mg_t_transp, indices))

        var.scatter_update(ops.IndexedSlices(var_t_values, indices))

        if self.stabilize is not None:
            self._stabilize(var)

    @def_function.function(experimental_compile=True)
    def _stabilize(self, var):
        if math_ops.floor_mod(self.iterations, self.stabilize) == 0:
            manifold = get_manifold(var)
            var.assign(manifold.projx(var))
            rms = self.get_slot(var, "rms")
            rms.assign(manifold.proju(var, rms))
            if self.centered:
                mg = self.get_slot(var, "mg")
                mg.assign(manifold.proju(var, mg))

    def set_weights(self, weights):
        params = self.weights
        if len(params) == len(weights) + 1:
            weights = [np.array(0)] + weights
        super(ConstrainedRMSprop, self).set_weights(weights)

    def get_config(self):
        config = super(ConstrainedRMSprop, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    "learning_rate"
                ),
                "decay": self._serialize_hyperparameter("decay"),
                "rho": self._serialize_hyperparameter("rho"),
                "epsilon": self.epsilon,
                "centered": self.centered,
                "stabilize": self.stabilize,
            }
        )
        return config
