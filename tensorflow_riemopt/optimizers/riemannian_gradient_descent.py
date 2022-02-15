"""Riemannian SGD optimizer implementation.

Bonnabel, Silvere. "Stochastic gradient descent on Riemannian manifolds."
IEEE Transactions on Automatic Control 58.9 (2013): 2217-2229.
"""
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


@generic_utils.register_keras_serializable(name="RiemannianSGD")
class RiemannianSGD(OptimizerV2):
    """Optimizer that implements the Riemannian SGD algorithm."""

    _HAS_AGGREGATE_GRAD = True

    def __init__(
        self,
        learning_rate=0.01,
        momentum=0.0,
        nesterov=False,
        stabilize=None,
        name="RiemannianSGD",
        **kwargs,
    ):
        """Construct a new Riemannian SGD optimizer.

        Bonnabel, Silvere. "Stochastic gradient descent on Riemannian
        manifolds." IEEE Transactions on Automatic Control 58.9 (2013):
        2217-2229.

        Args:
          learning_rate: A `Tensor`, floating point value, or a schedule that is a
            `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable that
            takes no arguments and returns the actual value to use, The learning
            rate. Defaults to 0.001.
          momentum: A float hyperparameter >= 0 that accelerates gradient descent
            in the relevant direction and dampens oscillations. Defaults to 0, i.e.,
            vanilla gradient descent.
          nesterov: boolean. Whether to apply Nesterov momentum. Defaults to `False`.
          stabilize: Project variables back to manifold every `stabilize` steps.
            Defaults to `None`.
          name: Optional name for the operations created when applying gradients.
            Defaults to "RiemannianSGD".
          **kwargs: Keyword arguments. Allowed to be one of `"clipnorm"` or
            `"clipvalue"`. `"clipnorm"` (float) clips gradients by norm; `"clipvalue"`
            (float) clips gradients by value.

        """

        super(RiemannianSGD, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._momentum = False
        if (
            isinstance(momentum, ops.Tensor)
            or callable(momentum)
            or momentum > 0
        ):
            self._momentum = True
        if isinstance(momentum, (int, float)) and (
            momentum < 0 or momentum > 1
        ):
            raise ValueError("`momentum` must be between [0, 1].")
        self._set_hyper("momentum", momentum)
        self.nesterov = nesterov
        self.stabilize = stabilize

    def _create_slots(self, var_list):
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(RiemannianSGD, self)._prepare_local(
            var_device, var_dtype, apply_state
        )
        apply_state[(var_device, var_dtype)]["momentum"] = array_ops.identity(
            self._get_hyper("momentum", var_dtype)
        )

    @def_function.function(experimental_compile=True)
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        manifold = get_manifold(var)
        grad = manifold.egrad2rgrad(var, grad)

        if self._momentum:
            momentum = self.get_slot(var, "momentum")
            momentum_t = momentum * self._momentum - grad * coefficients["lr_t"]
            if self.nesterov:
                var_t = manifold.retr(
                    var,
                    momentum_t * self._momentum - grad * coefficients["lr_t"],
                )
            else:
                var_t = manifold.retr(var, momentum_t)
            momentum.assign(manifold.transp(var, var_t, momentum_t))
            var.assign(var_t)
        else:
            var.assign(manifold.retr(var, -grad * coefficients["lr_t"]))

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

        var_values = array_ops.gather(var, indices)

        if self._momentum:
            momentum = self.get_slot(var, "momentum")
            momentum_t_values = (
                array_ops.gather(momentum, indices) * self._momentum
                - grad * coefficients["lr_t"]
            )
            if self.nesterov:
                var_t_values = manifold.retr(
                    var_values,
                    momentum_t_values * self._momentum
                    - grad * coefficients["lr_t"],
                )
            else:
                var_t_values = manifold.retr(var_values, momentum_t_values)
            momentum_transp_values = manifold.transp(
                var_values, var_t_values, momentum_t_values
            )
            momentum.scatter_update(
                ops.IndexedSlices(momentum_transp_values, indices)
            )
        else:
            var_t_values = manifold.retr(
                var_values, -grad * coefficients["lr_t"]
            )

        var.scatter_update(ops.IndexedSlices(var_t_values, indices))

        if self.stabilize is not None:
            self._stabilize(var)

    @def_function.function(experimental_compile=True)
    def _stabilize(self, var):
        if math_ops.floor_mod(self.iterations, self.stabilize) == 0:
            manifold = get_manifold(var)
            var.assign(manifold.projx(var))
            if self._momentum:
                momentum = self.get_slot(var, "momentum")
                momentum.assign(manifold.proju(var, momentum))

    def get_config(self):
        config = super(RiemannianSGD, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    "learning_rate"
                ),
                "decay": self._serialize_hyperparameter("decay"),
                "momentum": self._serialize_hyperparameter("momentum"),
                "nesterov": self.nesterov,
            }
        )
        return config
