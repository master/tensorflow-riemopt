import os

import tensorflow as tf

from tensorflow_riemopt.layers import ManifoldEmbedding
from tensorflow_riemopt.manifolds import Poincare
from tensorflow_riemopt.variable import get_manifold


class EmbeddingsTest(tf.test.TestCase):
    def test_layer(self):
        poincare = Poincare(k=5.0)
        with self.cached_session(use_gpu=True):
            inp = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int32)
            layer = ManifoldEmbedding(1000, 64, manifold=poincare)
            _ = layer(inp)
            self.assertEqual(
                type(get_manifold(layer.embeddings)), type(poincare)
            )
            model = tf.keras.Sequential(
                [tf.keras.Input((5,), dtype="int32"), layer]
            )
            _ = model(inp)
            path = os.path.join(self.get_temp_dir(), "model.keras")
            model.save(path)
            loaded = tf.keras.models.load_model(path)
            self.assertEqual(
                type(get_manifold(loaded.layers[-1].embeddings)), type(poincare)
            )
