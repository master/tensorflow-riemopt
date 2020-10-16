import numpy as np
import tensorflow as tf


def get_eps(val):
    return np.finfo(val.dtype.name).eps


def allclose(x, y, rtol=None, atol=None):
    """Return True if two arrays are element-wise equal within a tolerance."""
    rtol = 10 * get_eps(x) if rtol is None else rtol
    atol = 10 * get_eps(x) if atol is None else atol
    return tf.reduce_all(tf.abs(x - y) <= tf.abs(y) * rtol + atol)


def logm(inp):
    """Compute the matrix logarithm of positive-definite real matrices."""
    inp = tf.convert_to_tensor(inp)
    complex_dtype = tf.complex128 if inp.dtype == tf.float64 else tf.complex64
    log = tf.linalg.logm(tf.cast(inp, dtype=complex_dtype))
    return tf.cast(log, dtype=inp.dtype)


def transposem(inp):
    """Transpose multiple matrices."""
    perm = list(range(len(inp.shape)))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return tf.transpose(inp, perm)
