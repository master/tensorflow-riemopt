import tensorflow as tf
from tensorflow_riemopt.manifolds import Euclidean


def assign_to_manifold(var, manifold):
    if not hasattr(var, "shape"):
        raise ValueError(
            "var should be a valid variable with a 'shape' attribute"
        )
    if not manifold.check_shape(var):
        raise ValueError("Invalid variable shape {}".format(var.shape))
    setattr(var, "manifold", manifold)


def get_manifold(var, default_manifold=Euclidean()):
    if not hasattr(var, "shape"):
        raise ValueError(
            "var should be a valid variable with a 'shape' attribute"
        )
    return getattr(var, "manifold", default_manifold)
