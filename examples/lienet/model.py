import tensorflow as tf

from tensorflow_riemopt.variable import assign_to_manifold
from tensorflow_riemopt.manifolds import SpecialOrthogonal
from tensorflow_riemopt.manifolds import utils
from tensorflow_riemopt.optimizers import RiemannianSGD


EPS = 1e-7


def rot_angle(inputs):
    cos = (tf.linalg.trace(inputs) - 1) / 2.0
    return tf.math.acos(tf.clip_by_value(cos, -1.0, 1.0))


@tf.keras.utils.register_keras_serializable(name="RotMap")
class RotMap(tf.keras.layers.Layer):
    """Rotation Mapping layer."""

    def build(self, input_shape):
        """Create weights depending on the shape of the inputs.

        Expected `input_shape`:
        `[batch_size, spatial_dim, temp_dim, num_rows, num_cols]`, where
        `num_rows` = 3, `num_cols` = 3, `temp_dim` is the number of frames,
        and `spatial_dim` is the number of edges.
        """
        input_shape = input_shape.as_list()
        so = SpecialOrthogonal()
        self.w = self.add_weight(
            "w",
            shape=[input_shape[-4], input_shape[-2], input_shape[-1]],
            initializer=so.random,
        )
        assign_to_manifold(self.w, so)

    def call(self, inputs):
        return tf.einsum("sij,...stjk->...stik", self.w, inputs)


@tf.keras.utils.register_keras_serializable(name="RotPooling")
class RotPooling(tf.keras.layers.Layer):
    """Rotation Pooling layer."""

    def __init__(self, pooling, *args, **kwargs):
        """Instantiate the RotPooling layer.

        Args:
          pooling: `spatial` or `temporal` pooling type
        """
        super().__init__(*args, **kwargs)
        if not pooling in ["spatial", "temporal"]:
            raise ValueError("Invalid pooling type {}".format(pooling))
        self.pooling = pooling

    def call(self, inputs):
        if self.pooling == "spatial":
            stride = 2
            # reshape to [batch_size, temp_dim, spatial_dim, num_rows, num_cols]
            inputs = tf.transpose(inputs, perm=[0, 2, 1, 3, 4])
        elif self.pooling == "temporal":
            stride = 4
            # pad temp_dim to a multitude of stride
            temp_dim = tf.shape(inputs)[2]
            temp_pad = (
                tf.cast(tf.math.ceil(temp_dim / stride), tf.int32) * stride
                - temp_dim
            )
            padding = [[0, 0], [0, 0], [0, temp_pad], [0, 0], [0, 0]]
            inputs = tf.pad(inputs, padding)
        shape = tf.shape(inputs)
        new_shape = [
            shape[0],
            shape[1],
            shape[2] // stride,
            stride,
            shape[3],
            shape[4],
        ]
        inputs = tf.reshape(inputs, new_shape)
        thetas = rot_angle(inputs)
        indices = tf.math.argmax(thetas, axis=-1)
        results = tf.gather(inputs, indices, batch_dims=3)
        if self.pooling == "spatial":
            results = tf.transpose(results, perm=[0, 2, 1, 3, 4])
        return results

    def get_config(self):
        config = {"pooling": self.pooling}
        return dict(list(super().get_config().items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(name="LogMap")
class LogMap(tf.keras.layers.Layer):
    """Logarithmic Map layer."""

    def call(self, inputs):
        thetas = rot_angle(inputs)[..., tf.newaxis, tf.newaxis]
        zeros = tf.zeros_like(thetas)
        skew = (inputs - utils.transposem(inputs)) / 2.0
        log = skew * thetas / tf.math.sin(thetas)
        return tf.where(thetas < EPS, zeros, log)


def create_model(
    learning_rate,
    num_classes,
    pooling_types=["spatial", "temporal", "temporal"],
):
    """Instantiate the LieNet architecture.

    Huang, Zhiwu, et al. "Deep learning on Lie groups for skeleton-based
    action recognition." Proceedings of the IEEE conference on computer vision
    and pattern recognition. 2017.

    Args:
      learning_rate: model learning rate
      num_classes: number of output classes
      pooling_types: types of pooling layers in RotMap/RotPooling blocks
    """
    model = tf.keras.Sequential()
    for pooling in pooling_types:
        model.add(RotMap())
        model.add(RotPooling(pooling))
    model.add(LogMap())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(num_classes, use_bias=False))
    model.compile(
        optimizer=RiemannianSGD(learning_rate),
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )
    return model
