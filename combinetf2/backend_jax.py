import jax
import jax.numpy as jnp
from wums import logging

logger = logging.child_logger(__name__)

from combinetf2.backend import Backend


class BackendJax(Backend):

    _jnp_funcs = [
        "reciprocal",
        "diag",
        "eye",
        "where",
        "sqrt",
        "square",
        "concat",
        "matmul",
        "expand_dims",
    ]

    def __init__(self, options):
        jax.config.update("jax_enable_x64", True)

        super().__init__(options)
        # tf.config.experimental.enable_op_determinism()

        self.bn = jnp
        if options.eager:
            # TODO
            pass

        # tf.random.set_seed(options.seed)

        self._initialize_shared_funcs()

        for name in self._jnp_funcs:
            setattr(self, name, getattr(self.bn, name))

    @staticmethod
    def diag_part(*args, **kwargs):
        return jnp.diagonal(*args, **kwargs)

    @staticmethod
    def reduce_any(*args, **kwargs):
        return jnp.any(*args, **kwargs)

    @staticmethod
    def lu(*args, **kwargs):
        return jax.scipy.linalg.lu(*args, **kwargs)

    @staticmethod
    def variable(x, *args, **kwargs):
        # JAX has no mutable variables; just return x
        return x

    @staticmethod
    def assign(variable, values, *args, **kwargs):
        return values

    @staticmethod
    def maketensor(h5dset):
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
