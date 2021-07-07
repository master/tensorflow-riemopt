import tensorflow as tf

from tensorflow_riemopt.variable import assign_to_manifold
from tensorflow_riemopt.manifolds import StiefelEuclidean
from tensorflow_riemopt.manifolds import utils
from tensorflow_riemopt.optimizers import RiemannianSGD


@tf.keras.utils.register_keras_serializable(name="BiMap")
class BiMap(tf.keras.layers.Layer):
    """Bilinear Mapping layer."""

    def __init__(self, output_dim, *args, **kwargs):
        """Instantiate the BiMap layer.

        Args:
          output_dim: projection output dimension
        """
        super().__init__(*args, **kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.w = self.add_weight(
            "w", shape=[int(input_shape[-1]), self.output_dim]
        )
        assign_to_manifold(self.w, StiefelEuclidean())

    def call(self, inputs):
        return utils.transposem(self.w) @ inputs @ self.w

    def get_config(self):
        config = {"output_dim": self.output_dim}
        return dict(list(super().get_config().items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(name="ReEig")
class ReEig(tf.keras.layers.Layer):
    """Eigen Rectifier layer."""

    def __init__(self, epsilon=1e-4, *args, **kwargs):
        """Instantiate the ReEig layer.

        Args:
          epsilon: a rectification threshold value
        """
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        s, u, v = tf.linalg.svd(inputs)
        sigma = tf.maximum(s, self.epsilon)
        return u @ tf.linalg.diag(sigma) @ utils.transposem(v)

    def get_config(self):
        config = {"epsilon": self.epsilon}
        return dict(list(super().get_config().items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(name="LogEig")
class LogEig(tf.keras.layers.Layer):
    """Eigen Log layer."""

    def call(self, inputs):
        s, u, v = tf.linalg.svd(inputs)
        log_s = tf.math.log(s)
        return u @ tf.linalg.diag(log_s) @ utils.transposem(v)


def create_model(
    learning_rate, num_classes, bimap_dims=[200, 100, 50], eig_eps=1e-4
):
    """Instantiate the SPDNet architecture.

    Huang, Zhiwu, and Luc Van Gool. "A riemannian network for SPD matrix
    learning." Proceedings of the Thirty-First AAAI Conference on Artificial
    Intelligence. 2017.

    Args:
      learning_rate: model learning rate
      num_classes: number of output classes
      bimap_dims: dimensions of BiMap layers
      eig_eps: a rectification threshold value
    """

    model = tf.keras.Sequential()
    for output_dim in bimap_dims:
        model.add(BiMap(output_dim))
        model.add(ReEig(eig_eps))
    model.add(LogEig())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(num_classes, use_bias=False))
    model.compile(
        optimizer=RiemannianSGD(learning_rate),
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )
    return model
