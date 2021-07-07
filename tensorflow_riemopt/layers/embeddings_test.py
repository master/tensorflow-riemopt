import tensorflow as tf

from tensorflow_riemopt.layers import ManifoldEmbedding
from tensorflow_riemopt.manifolds import Grassmannian
from tensorflow_riemopt.variable import get_manifold


class EmbeddingsTest(tf.test.TestCase):
    def test_layer(self):
        grassmannian = Grassmannian()
        with self.cached_session(use_gpu=True):
            inp = tf.keras.Input((5, 2))
            layer = ManifoldEmbedding(1000, 64, manifold=grassmannian)
            _ = layer(inp)
            self.assertEqual(
                type(get_manifold(layer.embeddings)), type(grassmannian)
            )
            config = layer.get_config()
            from_config = ManifoldEmbedding(**config)
            _ = from_config(inp)
            self.assertEqual(
                type(get_manifold(from_config.embeddings)), type(grassmannian)
            )
