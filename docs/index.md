# TensorFlow RiemOpt

**TensorFlow RiemOpt** is a flexible, extensible library for Riemannian optimization and geometric deep learning in TensorFlow. It provides:
- Riemannian manifold classes with associated exponential and logarithmic maps, geodesics, and transports
- Riemannian optimizers (SGD, RMSProp, and adaptive methods such as Adam)
- Manifold-aware TensorFlow/Keras layers (e.g., Embedding)

## Installation

Install via PyPI:

```bash
pip install tensorflow-riemopt
```

## Quickstart

```{figure} _static/usage.png
:align: right
:width: 300px

Example: SPDNet classification on SPD manifold
```

```python
import tensorflow as tf
from tensorflow_riemopt.optimizers import ConstrainedRMSprop
from tensorflow_riemopt.manifolds import Grassmannian
from tensorflow_riemopt.variable import assign_to_manifold

# Create a variable on the Grassmannian manifold (3×2 matrix)
manifold = Grassmannian()
x = tf.Variable(tf.random.uniform((3, 2)), dtype=tf.float32)
assign_to_manifold(x, manifold)

# Define a simple loss (squared Frobenius norm)
with tf.GradientTape() as tape:
    loss = tf.reduce_sum(x * x)

# Compute gradients and update
grads = tape.gradient(loss, [x])
opt = ConstrainedRMSprop(learning_rate=0.1, rho=0.9)
opt.apply_gradients(zip(grads, [x]))
```

## Documentation

```{toctree}
:titlesonly:
:glob:
:maxdepth: 1

api/*
```

## Examples

The repository provides several fully implemented example network projects:

- **GrNet**: Deep networks on Grassmann manifolds. [examples/grnet](https://github.com/master/tensorflow-riemopt/tree/master/examples/grnet)
- **LieNet**: Deep learning on Lie groups for action recognition. [examples/lienet](https://github.com/master/tensorflow-riemopt/tree/master/examples/lienet)
- **SPDNet**: Riemannian network for SPD matrix learning. [examples/spdnet](https://github.com/master/tensorflow-riemopt/tree/master/examples/spdnet)
