import tensorflow as tf

from tensorflow_riemopt.variable import assign_to_manifold
from tensorflow_riemopt.manifolds import Manifold


@tf.keras.utils.register_keras_serializable(name="ManifoldEmbedding")
class ManifoldEmbedding(tf.keras.layers.Embedding):
    def __init__(self, *args, manifold, **kwargs):
        super().__init__(*args, **kwargs)
        self.manifold = manifold

    def build(self, input_shape):
        super().build(input_shape)
        assign_to_manifold(self.embeddings, self.manifold)

    def get_config(self):
        config = {
            "manifold": tf.keras.utils.serialize_keras_object(self.manifold)
        }
        return dict(list(super().get_config().items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        manifold = tf.keras.utils.deserialize_keras_object(
            config.pop("manifold")
        )
        return cls(manifold=manifold, **config)
