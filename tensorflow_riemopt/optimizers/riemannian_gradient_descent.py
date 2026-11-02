"""Riemannian SGD optimizer implementation.

Bonnabel, Silvere. "Stochastic gradient descent on Riemannian manifolds."
IEEE Transactions on Automatic Control 58.9 (2013): 2217-2229.
"""

import keras
import tensorflow as tf

from tensorflow_riemopt.variable import get_manifold


def _rows(gradient, variable):
    if isinstance(gradient, tf.IndexedSlices):
        indices, segments = tf.unique(gradient.indices)
        grad = tf.math.unsorted_segment_sum(
            gradient.values, segments, tf.shape(indices)[0]
        )
        gather = lambda t: tf.gather(t, indices)
        scatter = lambda t: tf.IndexedSlices(t, indices)
    else:
        grad, gather, scatter = gradient, tf.convert_to_tensor, tf.identity
    return tf.cast(grad, variable.dtype), gather, scatter


@keras.saving.register_keras_serializable(
    package="tensorflow_riemopt", name="RiemannianSGD"
)
class RiemannianSGD(keras.optimizers.SGD):
    """Optimizer that implements the Riemannian SGD algorithm."""

    def __init__(self, *args, stabilize=None, **kwargs):
        """Construct a new Riemannian SGD optimizer.

        Bonnabel, Silvere. "Stochastic gradient descent on Riemannian
        manifolds." IEEE Transactions on Automatic Control 58.9 (2013):
        2217-2229.

        Args:
          stabilize: Project variables back to manifold every `stabilize` steps.
            Defaults to `None`.
          **kwargs: Keyword arguments of `keras.optimizers.SGD`. `weight_decay`
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
        x = gather(variable)
        rgrad = manifold.egrad2rgrad(x, grad)
        if self.momentum != 0:
            m = self.momentums[self._get_variable_index(variable)]
            momentum = tf.cast(self.momentum, variable.dtype)
            m_t = gather(m) * momentum - rgrad * lr
            if self.nesterov:
                x_t = manifold.retr(x, m_t * momentum - rgrad * lr)
            else:
                x_t = manifold.retr(x, m_t)
            self.assign(m, scatter(manifold.transp(x, x_t, m_t)))
        else:
            x_t = manifold.retr(x, -rgrad * lr)
        self.assign(variable, scatter(x_t))
        if self.stabilize is not None:
            self._stabilize(variable, manifold)

    def _stabilize(self, variable, manifold):
        def project():
            self.assign(variable, manifold.projx(variable))
            if self.momentum != 0:
                m = self.momentums[self._get_variable_index(variable)]
                self.assign(
                    m, manifold.proju(variable, tf.convert_to_tensor(m))
                )

        tf.cond(
            tf.equal(tf.math.floormod(self.iterations, self.stabilize), 0),
            project,
            lambda: None,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"stabilize": self.stabilize})
        return config
