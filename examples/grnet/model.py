import tensorflow as tf

from tensorflow_riemopt.variable import assign_to_manifold
from tensorflow_riemopt.manifolds import Grassmannian
from tensorflow_riemopt.manifolds import utils
from tensorflow_riemopt.optimizers import RiemannianSGD


@tf.keras.utils.register_keras_serializable(name="FRMap")
class FRMap(tf.keras.layers.Layer):
    """Full Rank Mapping layer."""

    def __init__(self, output_dim, num_proj=8, *args, **kwargs):
        """Instantiate the FRMap layer.

        Args:
          output_dim: projection output dimension
          num_proj: number of projections to compute
        """
        super().__init__(*args, **kwargs)
        self.output_dim = output_dim
        self.num_proj = num_proj

    def build(self, input_shape):
        grassmannian = Grassmannian()
        self.w = self.add_weight(
            "w",
            shape=[self.num_proj, input_shape[-2], self.output_dim],
            initializer=grassmannian.random,
        )
        assign_to_manifold(self.w, grassmannian)
        self._expand = len(input_shape) == 3

    def call(self, inputs):
        if self._expand:
            inputs = tf.expand_dims(inputs, -3)
        return utils.transposem(self.w) @ inputs

    def get_config(self):
        config = {"output_dim": self.output_dim, "num_proj": self.num_proj}
        return dict(list(super().get_config().items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(name="ReOrth")
class ReOrth(tf.keras.layers.Layer):
    """Re-Orthonormalization layer."""

    def call(self, inputs):
        q, _r = tf.linalg.qr(inputs)
        return q


@tf.keras.utils.register_keras_serializable(name="ProjMap")
class ProjMap(tf.keras.layers.Layer):
    """Projection Mapping layer."""

    def call(self, inputs):
        return inputs @ utils.transposem(inputs)


@tf.keras.utils.register_keras_serializable(name="ProjPooling")
class ProjPooling(tf.keras.layers.Layer):
    """Projection Pooling layer."""

    def __init__(self, stride=2, *args, **kwargs):
        """Instantiate the ProjPooling layer.

        Args:
          stride: factor by which to downscale
        """
        super().__init__(*args, **kwargs)
        self.stride = stride

    def call(self, inputs):
        shape = tf.shape(inputs)
        new_shape = [
            shape[0],
            shape[1],
            self.stride,
            shape[2] // self.stride,
            shape[3],
        ]
        return tf.reduce_mean(
            tf.reshape(inputs, new_shape), axis=-3, keepdims=False
        )

    def get_config(self):
        config = {"stride": self.stride}
        return dict(list(super().get_config().items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(name="OrthMap")
class OrthMap(tf.keras.layers.Layer):
    """Orthonormal Mapping layer."""

    def __init__(self, top_eigen, *args, **kwargs):
        """Instantiate the OrthMap layer.

        Args:
          num_eigen: number of top eigenvectors to retain
        """
        super().__init__(*args, **kwargs)
        self.top_eigen = top_eigen

    def call(self, inputs):
        _s, u, _vt = tf.linalg.svd(inputs)
        return u[..., : self.top_eigen]

    def get_config(self):
        config = {"top_eigen": self.top_eigen}
        return dict(list(super().get_config().items()) + list(config.items()))


def create_model(
    learning_rate,
    num_classes,
    frmap_dims=[300, 100],
    pool_stride=2,
    top_eigen=10,
):
    """Instantiate the GrNet architecture.

    Huang, Zhiwu, Jiqing Wu, and Luc Van Gool. "Building Deep Networks on
    Grassmann Manifolds." AAAI. AAAI Press, 2018.

    Args:
      learning_rate: model learning rate
      num_classes: number of output classes
      frmap_dims: dimensions of FrMap layers
      pool_stride: pooling stride
      top_eigen: number of eigenvectors to retain in OrthMap
    """
    model = tf.keras.Sequential()
    for output_dim in frmap_dims:
        model.add(FRMap(output_dim))
        model.add(ReOrth())
        model.add(ProjMap())
        model.add(ProjPooling(pool_stride))
        model.add(OrthMap(top_eigen))
    model.add(ProjMap())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(num_classes, use_bias=False))
    model.compile(
        optimizer=RiemannianSGD(learning_rate),
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )
    return model
