# TensorFlow RiemOpt

```{figure} _static/usage.png
:align: right
:width: 300px
```

**TensorFlow RiemOpt** is a flexible, extensible library for Riemannian optimization and geometric deep learning in TensorFlow. It provides:
- Riemannian manifold classes with associated exponential and logarithmic maps, geodesics, and transports
- Riemannian optimizers (SGD, RMSProp, and adaptive methods such as Adam)
- Manifold-aware TensorFlow/Keras layers (e.g., Embedding)

## Installation

Install via PyPI:

```bash
pip install tensorflow-riemopt
```

Or add it to a project managed by uv:

```bash
uv add tensorflow-riemopt
```

## Quickstart

```python
import tensorflow as tf
from tensorflow_riemopt.optimizers import ConstrainedRMSprop
from tensorflow_riemopt.manifolds import Grassmannian
from tensorflow_riemopt.variable import assign_to_manifold

# Create a variable on the Grassmannian manifold (3×2 matrix)
manifold = Grassmannian()
x = tf.Variable(manifold.random((3, 2)))
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

- **ANTHEM**: Choudhary, Nurendra, Nikhil Rao, Sumeet Katariya, Karthik Subbian, and Chandan K. Reddy. "ANTHEM: Attentive Hyperbolic Entity Model for Product Search." In Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining, 2022. [https://github.com/amazon-science/hyperbolic-embeddings/tree/main/product_matching](https://github.com/amazon-science/hyperbolic-embeddings/tree/main/product_matching)
- **GrNet**: Deep networks on Grassmann manifolds. [examples/grnet](https://github.com/master/tensorflow-riemopt/tree/master/examples/grnet)
- **LieNet**: Deep learning on Lie groups for action recognition. [examples/lienet](https://github.com/master/tensorflow-riemopt/tree/master/examples/lienet)
- **SPDNet**: Riemannian network for SPD matrix learning. [examples/spdnet](https://github.com/master/tensorflow-riemopt/tree/master/examples/spdnet)

## Contributing

Report bugs and request features in the [issue tracker](https://github.com/master/tensorflow-riemopt/issues) along with a minimal snippet that reproduces the problem. To work on the code, clone the repository and run:

```bash
uv sync --locked --group dev
uv run pytest -q tensorflow_riemopt
uv run black --line-length 80 --check tensorflow_riemopt
```

## Contact

For support or development inquiries, contact us at: [tensorflow-riemopt-dev@googlegroups.com](mailto:tensorflow-riemopt-dev@googlegroups.com)

## Citation

If you find TensorFlow RiemOpt useful in your research, please cite:

```bibtex
@misc{smirnov2021tensorflow,
      title={TensorFlow RiemOpt: a library for optimization on Riemannian manifolds},
      author={Oleg Smirnov},
      year={2021},
      eprint={2105.13921},
      archivePrefix={arXiv},
      primaryClass={cs.MS}
}
```
