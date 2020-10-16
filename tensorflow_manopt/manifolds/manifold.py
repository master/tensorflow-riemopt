import tensorflow as tf
import abc


class Manifold(metaclass=abc.ABCMeta):
    name = "Base"
    ndims = None

    def __repr__(self):
        """Returns a string representation of the particular manifold."""
        return "{0} (ndims={1}) manifold".format(self.name, self.ndims)

    def check_shape(self, shape_or_tensor):
        """Check if given shape is compatible with the manifold."""
        shape = (
            shape_or_tensor.shape
            if hasattr(shape_or_tensor, "shape")
            else shape_or_tensor
        )
        return (len(shape) >= self.ndims) & self._check_shape(shape)

    def check_point_on_manifold(self, x, atol=None, rtol=None):
        """Check if point :math:`x` lies on the manifold."""
        return self.check_shape(x) & self._check_point_on_manifold(
            x, atol=atol, rtol=rtol
        )

    def check_vector_on_tangent(self, x, u, atol=None, rtol=None):
        """Check if vector :math:`u` lies on the tangent space at :math:`x`."""
        return (
            self._check_point_on_manifold(x, atol=atol, rtol=rtol)
            & self.check_shape(u)
            & self._check_vector_on_tangent(x, u, atol=atol, rtol=rtol)
        )

    def _check_shape(self, shape):
        return tf.constant(True)

    @abc.abstractmethod
    def _check_point_on_manifold(self, x, atol, rtol):
        raise NotImplementedError

    @abc.abstractmethod
    def _check_vector_on_tangent(self, x, u, atol, rtol):
        raise NotImplementedError

    @abc.abstractmethod
    def dist(self, x, y, keepdims=False):
        """Compute the distance between two points :math:`x` and :math:`y: along a
        geodesic.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def inner(self, x, u, v, keepdims=False):
        """Return the inner product (i.e., the Riemannian metric) between two tangent
        vectors :math:`u` and :math:`v` in the tangent space at :math:`x`.
        """
        raise NotImplementedError

    def norm(self, x, u, keepdims=False):
        """Compute the norm of a tangent vector :math:`u` in the tangent space at
        :math:`x`.
        """
        return self.inner(x, u, u, keepdims=keepdims) ** 0.5

    @abc.abstractmethod
    def proju(self, x, u):
        """Project a vector :math:`u` in the ambient space on the tangent space at
        :math:`x`
        """
        raise NotImplementedError

    def egrad2rgrad(self, x, u):
        """Map the Euclidean gradient :math:`u` in the ambient space on the tangent
        space at :math:`x`.
        """
        return self.proju(x, u)

    @abc.abstractmethod
    def projx(self, x):
        """Project a point :math:`x` on the manifold."""
        raise NotImplementedError

    @abc.abstractmethod
    def retr(self, x, u):
        """Perform a retraction from point :math:`x` with given direction :math:`u`."""
        raise NotImplementedError

    @abc.abstractmethod
    def exp(self, x, u):
        r"""Perform an exponential map :math:`\operatorname{Exp}_x(u)`."""
        raise NotImplementedError

    @abc.abstractmethod
    def log(self, x, y):
        r"""Perform a logarithmic map :math:`\operatorname{Log}_{x}(y)`."""
        raise NotImplementedError

    @abc.abstractmethod
    def transp(self, x, y, v):
        r"""Perform a vector transport :math:`\mathfrak{T}_{x\to y}(v)`."""
        return NotImplementedError

    def ptransp(self, x, y, v):
        r"""Perform a parallel transport :math:`\operatorname{P}_{x\to y}(v)`."""
        return NotImplementedError

    def random(self, shape, dtype=tf.float32):
        """Sample a random point on the manifold."""
        return self.projx(tf.random.uniform(shape, dtype=dtype))

    def geodesic(self, x, u, t):
        """Geodesic from point :math:`x` in the direction of tanget vector
        :math:`u`
        """
        raise NotImplementedError

    def pairmean(self, x, y):
        """Compute a Riemannian (Fréchet) mean of points :math:`x` and :math:`y`"""
        return self.geodesic(x, self.log(x, y), 0.5)
