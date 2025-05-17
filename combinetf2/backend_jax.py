import jax
import jax.numpy as jnp
from wums import logging

logger = logging.child_logger(__name__)

from combinetf2.backend import Backend


class BackendJax(Backend):
    def __init__(self, options):
        super().__init__(options)
        # tf.config.experimental.enable_op_determinism()

        if options.eager:
            # TODO
            pass

        # tf.random.set_seed(options.seed)

    def ones(self, *args, **kwargs):
        return jnp.ones(*args, **kwargs)

    def ones_like(self, *args, **kwargs):
        return jnp.ones_like(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        return jnp.zeros(*args, **kwargs)

    def zeros_like(self, *args, **kwargs):
        return jnp.zeros_like(*args, **kwargs)

    def eye(self, *args, **kwargs):
        return jnp.eye(*args, **kwargs)

    def sqrt(self, *args, **kwargs):
        return jnp.sqrt(*args, **kwargs)

    def square(self, *args, **kwargs):
        return jnp.square(*args, **kwargs)

    def reciprocal(self, *args, **kwargs):
        return jnp.reciprocal(*args, **kwargs)

    def concat(self, *args, **kwargs):
        return jnp.concatenate(*args, **kwargs)

    def reduce_any(self, *args, **kwargs):
        return jnp.any(*args, **kwargs)

    def where(self, *args, **kwargs):
        return jnp.where(*args, **kwargs)

    def diag_part(self, *args, **kwargs):
        return jnp.diag_part(*args, **kwargs)

    def diag(self, *args, **kwargs):
        return jnp.diag(*args, **kwargs)

    def diag_operator(self, *args, **kwargs):
        return jnp.diag(*args, **kwargs)

    def lu(self, *args, **kwargs):
        return jax.scipy.linalg.lu(*args, **kwargs)

    def variable(self, x, *args, **kwargs):
        # JAX has no mutable variables; just return x
        return x

    def maketensor(self, h5dset):
        # Determine the intended shape
        shape = h5dset.attrs.get("original_shape", h5dset.shape)

        # Handle empty tensors
        if h5dset.size == 0:
            return jnp.zeros(shape, dtype=h5dset.dtype)

        # Load from HDF5 directly into a NumPy array
        data_np = h5dset[...]

        # Reshape if the dataset was flattened
        data_np = data_np.reshape(shape)

        # Convert to JAX array
        return jnp.array(data_np)
