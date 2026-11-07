---
title: 'tensorflow-riemopt: a library for optimization on Riemannian manifolds in TensorFlow'
tags:
  - Python
  - TensorFlow
  - Riemannian geometry
  - manifold optimization
  - geometric deep learning
authors:
  - name: Oleg Smirnov
    corresponding: true
    affiliation: 1
affiliations:
  - name: Microsoft Gaming, Stockholm, Sweden
    index: 1
date: 30 October 2026
bibliography: paper.bib
---

# Summary

tensorflow-riemopt is a TensorFlow [@abadi2016tensorflow] library for training
machine learning models whose parameters are constrained to Riemannian
manifolds. It provides multiple manifolds with their Riemannian metrics,
exponential and logarithmic maps, geodesics, retractions and vector
transports, implemented either as closed-form expressions or approximations
where applicable. The library also implements Riemannian versions of
stochastic gradient descent [@bonnabel2013stochastic] and of the adaptive
RMSprop [@roy2018geometry] and Adam [@becigneul2018riemannian]
algorithms. After a manifold is attached to a TensorFlow variable, Keras
layers and training loops can be used without modification, resulting in a
geometry-aware model that can be trained and served inside a TensorFlow
Extended pipeline [@baylor2017tfx].

The repository includes example implementations of SPDNet
[@huang2017riemannian], GrNet [@huang2018building], and LieNet
[@huang2017deep] methods built on the library.

# Statement of need

Several families of machine learning models keep their parameters on a
non-Euclidean space. Hyperbolic embeddings represent data with latent
hierarchies and are used in language modeling [@nickel2017poincare;
@nickel2018learning; @sala2018representation; @ganea2018hyperbolic],
recommendation [@chamberlain2019scalable] and image classification
[@khrulkov2020hyperbolic]. Weights on the Stiefel and Grassmann manifolds
appear in networks for emotion and action recognition from video
[@huang2017riemannian; @huang2018building; @chakraborty2019statistics] and in
image set classification [@li2015face]. Symmetric positive definite matrices
represent diffusion tensors and covariance descriptors
[@pennec2006riemannian], and the Lie groups $\mathrm{SO}(3)$ and
$\mathrm{SE}(3)$ parameterize pose estimation and skeleton-based action
recognition [@hou2018computing; @huang2017deep]. Training of these models
requires an optimizer whose updates are constrained to the manifold, which is
not guaranteed by the conventional Euclidean optimizers.

# State of the field

Riemannian stochastic optimization [@bonnabel2013stochastic; @roy2018geometry;
@becigneul2018riemannian; @kasai2019riemannian] has support in PyTorch through
Geoopt [@kochurov2020geoopt] and McTorch [@meghwanshi2018mctorch]. TensorFlow,
and the related TFX and TensorFlow Serving frameworks, had no maintained
equivalent. tensorflow-riemopt closes this gap by providing
manifold-constrained variables and Riemannian optimizers that replace their
Euclidean counterparts in any existing TensorFlow / Keras pipeline. It also
exposes a low-level API for the geometric operations, so new manifolds and
algorithms are implemented and tested against the existing alternatives.

Across related work, Geoopt [@kochurov2020geoopt] and McTorch
[@meghwanshi2018mctorch] implement Riemannian stochastic optimizers for
PyTorch, while Pymanopt [@townsend2016pymanopt] implements deterministic
solvers and differentiates cost functions supported by TensorFlow, PyTorch,
and JAX backends. However, Pymanopt computes the manifold operations in NumPy
and does not provide stochastic optimizers, so it does not support minibatch
training of neural networks. Geomstats [@miolane2020geomstats] implements
additional manifolds and differential-geometric constructions, providing a
scikit-learn API [@pedregosa2011scikit] with NumPy, PyTorch, and Autograd
backends. Manopt [@boumal2014manopt], Manifolds.jl [@axen2023manifolds] and
Rieoptax [@utpala2022rieoptax] provide alternative implementations in MATLAB,
Julia, and JAX correspondingly.

# Software design

In tensorflow-riemopt, the manifold constraint is a property of the variable,
that can be set by `assign_to_manifold(variable, manifold)`
function. Therefore, a model can combine manifold-constrained and
unconstrained layers, which are treated accordingly by the Riemannian
optimizers. After training, the exported model contains only standard
TensorFlow operations, so TensorFlow TFX components do not depend on the
library.

```python
import tensorflow as tf
from tensorflow_riemopt.variable import assign_to_manifold
from tensorflow_riemopt.manifolds import StiefelEuclidean
from tensorflow_riemopt.optimizers import RiemannianSGD


class BiMap(tf.keras.layers.Layer):
    """Bilinear mapping layer of SPDNet with a weight on the Stiefel manifold."""

    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.w = self.add_weight(name="w", shape=[input_shape[-1], self.output_dim])
        assign_to_manifold(self.w, StiefelEuclidean())

    def call(self, inputs):
        return tf.transpose(self.w) @ inputs @ self.w


model = tf.keras.Sequential([BiMap(32), tf.keras.layers.Flatten(), tf.keras.layers.Dense(7)])
model.compile(optimizer=RiemannianSGD(learning_rate=0.01), loss="sparse_categorical_crossentropy")
```

As in Manopt [@boumal2014manopt], each manifold subclasses `Manifold` and
implements the methods the optimization algorithms call: the Riemannian metric
`inner(x, u, v)`, the tangent-space projection `proju(x, u)`, a retraction
`retr(x, u)` and a vector transport `transp(x, y, v)`. Exponential and
logarithmic maps and parallel transport are additional methods that can be
added for adaptive optimization support.

```python
from tensorflow_riemopt.manifolds import Sphere

S = Sphere()
x = S.projx(tf.constant([0.1, -0.1, 0.1]))
u = S.proju(x, tf.constant([1.0, 1.0, 1.0]))
v = S.proju(x, tf.constant([-0.7, -1.4, 1.4]))
y = S.exp(x, v)
u_y = S.transp(x, y, u)
v_y = S.transp(x, y, v)
```

![Geometric operations on the sphere $\mathbb{S}^2$. The exponential map takes
 $x$ along $v$ to $y$, and vector transport carries $u$ and $v$ from
 $T_x\mathcal{M}$ to $T_y\mathcal{M}$.\label{fig:sphere}](examples/usage.png){
 width=60% }

The library implements the exponential and logarithmic map in closed form
where it is known. Otherwise, it falls back to a published approximate
algorithm, e.g., the matrix-algebraic logarithm on the Stiefel manifold under
the canonical metric [@zimmermann2017matrix] and the Cayley retraction
[@li2020efficient]. Every manifold method is composed of TensorFlow
operations, so automatic differentiation, graph execution and eager execution
do not require manifold-specific code.

The optimizers implement the Keras optimizer interface, including the sparse
update path for `tf.IndexedSlices`, so in an embedding table on the Poincaré
ball or the hyperboloid only the rows present in a batch are updated. The
`layers` module provides such a table as `ManifoldEmbedding`. Optimizer state
and hyperparameters are serialized through the Keras configuration mechanism,
so a compiled model is saved and restored with the standard calls.

# Research impact statement

As of 30 October 2026, Google Scholar lists 12 works citing the preprint
[@smirnov2021tensorflow], the GitHub repository has ~100 stars and ~20 forks.
ANTHEM [@choudhary2022anthem], a hyperbolic entity model for product search,
imports the library in the public implementation released by Amazon Science,
which installs it from our repository under its former name
tensorflow-manopt. The monograph on Riemannian geometry and geometric
statistics by @guigui2023introduction and the survey of geometric optimization
for deep learning by @fei2025survey cite the library, and @axen2023manifolds
include it in their cross-library benchmark of manifold operations. The citing
research includes Riemannian Hamiltonian methods for min-max optimization
[@han2023riemannian], drug-target interaction prediction [@binatli2023mokpe]
and differentially private Fréchet means on symmetric positive definite
matrices [@utpala2022differentially].

The package is distributed on PyPI, documented on Read the Docs, and tested in
continuous integration with coverage reporting against current TensorFlow
releases.

# AI usage disclosure

A generative AI tool was used to prepare the manuscript draft from an earlier
preprint describing the library [@smirnov2021tensorflow]. The final draft was
reviewed and edited to verify its claims against the repository and the cited
works.

# Acknowledgements

We thank the users who reported issues and contributed fixes to the
repository.

# References
