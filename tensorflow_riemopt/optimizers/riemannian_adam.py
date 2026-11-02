"""Riemannian Adam optimizer implementation.

Becigneul, Gary, and Octavian-Eugen Ganea. "Riemannian Adaptive Optimization
Methods." International Conference on Learning Representations. 2018.
"""

import keras
import tensorflow as tf

from tensorflow_riemopt.optimizers.riemannian_gradient_descent import _rows
from tensorflow_riemopt.variable import get_manifold


@keras.saving.register_keras_serializable(
    package="tensorflow_riemopt", name="RiemannianAdam"
)
class RiemannianAdam(keras.optimizers.Adam):
    """Optimizer that implements the Riemannian Adam algorithm."""

    def __init__(self, *args, stabilize=None, **kwargs):
        """Construct a new Riemannian Adam optimizer.

        Becigneul, Gary, and Octavian-Eugen Ganea. "Riemannian Adaptive
        Optimization Methods." International Conference on Learning
        Representations. 2018.

        Args:
          stabilize: Project variables back to manifold every `stabilize` steps.
            Defaults to `None`.
          **kwargs: Keyword arguments of `keras.optimizers.Adam`. `weight_decay`
            and `use_ema` are not supported.

        """
        super().__init__(*args, **kwargs)
        if self.weight_decay is not None:
            raise ValueError("weight_decay is not supported on manifolds")
        if self.use_ema:
            raise ValueError("use_ema is not supported on manifolds")
        self.stabilize = stabilize

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
        manifold = get_manifold(
            self._trainable_variables[self._get_variable_index(variable)]
        )
        grad, gather, scatter = _rows(gradient, variable)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)
        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        i = self._get_variable_index(variable)
        m, v = self._momentums[i], self._velocities[i]
        x = gather(variable)
        rgrad = manifold.egrad2rgrad(x, grad)
        m_t = gather(m) + (rgrad - gather(m)) * (1 - self.beta_1)
        v_t = gather(v) + (
            manifold.inner(x, rgrad, rgrad, keepdims=True) - gather(v)
        ) * (1 - self.beta_2)
        self.assign(v, scatter(v_t))
        if self.amsgrad:
            vhat = self._velocity_hats[i]
            v_t = tf.maximum(gather(vhat), v_t)
            self.assign(vhat, scatter(v_t))
        x_t = manifold.retr(x, -(m_t * alpha) / (tf.sqrt(v_t) + self.epsilon))
        self.assign(m, scatter(manifold.transp(x, x_t, m_t)))
        self.assign(variable, scatter(x_t))
        if self.stabilize is not None:
            self._stabilize(variable, manifold)

    def _stabilize(self, variable, manifold):
        def project():
            self.assign(variable, manifold.projx(variable))
            m = self._momentums[self._get_variable_index(variable)]
            self.assign(m, manifold.proju(variable, tf.convert_to_tensor(m)))

        tf.cond(
            tf.equal(tf.math.floormod(self.iterations, self.stabilize), 0),
            project,
            lambda: None,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"stabilize": self.stabilize})
        return config
