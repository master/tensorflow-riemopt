import tensorflow as tf

from tensorflow_riemopt.variable import assign_to_manifold
from tensorflow_riemopt.manifolds import Manifold


@tf.keras.utils.register_keras_serializable(name="ManifoldEmbedding")
class ManifoldEmbedding(tf.keras.layers.Embedding):
    def __init__(self, *args, manifold, **kwargs):
        super().__init__(*args, **kwargs)
        self.manifold = manifold
        self._embeddings_initializer = self.embeddings_initializer
        initializer = self._embeddings_initializer

        def embeddings_initializer(shape, dtype=None):
            return manifold.projx(initializer(shape, dtype=dtype))

        self.embeddings_initializer = embeddings_initializer

    def build(self, input_shape):
        super().build(input_shape)
        assign_to_manifold(self.embeddings, self.manifold)

    def get_config(self):
        config = {
            "manifold": tf.keras.utils.serialize_keras_object(self.manifold),
            "embeddings_initializer": tf.keras.initializers.serialize(
                self._embeddings_initializer
            ),
        }
        return dict(list(super().get_config().items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        manifold = tf.keras.utils.deserialize_keras_object(
            config.pop("manifold")
        )
        return cls(manifold=manifold, **config)
