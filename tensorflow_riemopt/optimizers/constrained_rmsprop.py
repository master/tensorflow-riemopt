"""Constrained RMSprop optimizer implementation.

Kumar Roy, Soumava, Zakaria Mhammedi, and Mehrtash Harandi. "Geometry aware
constrained optimization techniques for deep learning." Proceedings of the
IEEE Conference on Computer Vision and Pattern Recognition. 2018.
"""

import keras
import tensorflow as tf

from tensorflow_riemopt.optimizers.riemannian_gradient_descent import _rows
from tensorflow_riemopt.variable import get_manifold


@keras.saving.register_keras_serializable(
    package="tensorflow_riemopt", name="ConstrainedRMSprop"
)
class ConstrainedRMSprop(keras.optimizers.RMSprop):
    """Optimizer that implements the RMSprop algorithm."""

    def __init__(self, *args, stabilize=None, **kwargs):
        """Construct a new Constrained RMSprop optimizer.

        Kumar Roy, Soumava, Zakaria Mhammedi, and Mehrtash Harandi. "Geometry
        aware constrained optimization techniques for deep learning." Proceedings
        of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.

        Args:
          stabilize: Project variables back to manifold every `stabilize` steps.
            Defaults to `None`.
          **kwargs: Keyword arguments of `keras.optimizers.RMSprop`.
            `weight_decay`, `use_ema` and a nonzero `momentum` are not
            supported.
        """
        super().__init__(*args, **kwargs)
        if self.weight_decay is not None:
            raise ValueError("weight_decay is not supported on manifolds")
        if self.use_ema:
            raise ValueError("use_ema is not supported on manifolds")
        if self.momentum != 0:
            raise ValueError("momentum is not supported by ConstrainedRMSprop")
        self.stabilize = stabilize

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        manifold = get_manifold(
            self._trainable_variables[self._get_variable_index(variable)]
        )
        grad, gather, scatter = _rows(gradient, variable)
        i = self._get_variable_index(variable)
        rms = self._velocities[i]
        x = gather(variable)
        rgrad = manifold.egrad2rgrad(x, grad)
        rms_t = self.rho * gather(rms) + (1 - self.rho) * rgrad * rgrad
        denom_t = rms_t
        if self.centered:
            mg = self._average_gradients[i]
            mg_t = self.rho * gather(mg) + (1 - self.rho) * rgrad
            denom_t = denom_t - tf.square(mg_t)
            self.assign(mg, scatter(mg_t))
        step = manifold.proju(x, -lr * rgrad / tf.sqrt(denom_t + self.epsilon))
        x_t = manifold.retr(x, step)
        self.assign(rms, scatter(rms_t))
        self.assign(variable, scatter(x_t))
        if self.stabilize is not None:
            self._stabilize(variable, manifold)

    def _stabilize(self, variable, manifold):
        tf.cond(
            tf.equal(tf.math.floormod(self.iterations, self.stabilize), 0),
            lambda: self.assign(variable, manifold.projx(variable)),
            lambda: None,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"stabilize": self.stabilize})
        return config
