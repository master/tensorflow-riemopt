import tensorflow as tf

from tensorflow_riemopt.manifolds import Euclidean


def assign_to_manifold(var, manifold):
    if not isinstance(var, tf.Variable):
        raise ValueError("var should be a TensorFlow variable")
    if not manifold.check_shape(var):
        raise ValueError("Invalid variable shape {}".format(var.shape))
    setattr(var, "manifold", manifold)


def get_manifold(var, default_manifold=Euclidean()):
    if not isinstance(var, tf.Variable):
        raise ValueError("var should be a TensorFlow variable")
    if hasattr(var, "manifold"):
        return var.manifold
    else:
        return default_manifold
