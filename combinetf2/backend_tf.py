import tensorflow as tf
from wums import logging

from combinetf2.backend import Backend

logger = logging.child_logger(__name__)


class BackendTf(Backend):
    def __init__(self, options):
        super().__init__(options)
        tf.config.experimental.enable_op_determinism()

        if options.eager:
            tf.config.run_functions_eagerly(True)

        tf.random.set_seed(options.seed)

    def ones(self, *args, **kwargs):
        return tf.ones(*args, **kwargs)

    def ones_like(self, *args, **kwargs):
        return tf.ones_like(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        return tf.zeros(*args, **kwargs)

    def zeros_like(self, *args, **kwargs):
        return tf.zeros_like(*args, **kwargs)

    def eye(self, *args, **kwargs):
        return tf.eye(*args, **kwargs)

    def sqrt(self, *args, **kwargs):
        return tf.sqrt(*args, **kwargs)

    def square(self, *args, **kwargs):
        return tf.square(*args, **kwargs)

    def reciprocal(self, *args, **kwargs):
        return tf.math.reciprocal(*args, **kwargs)

    def concat(self, *args, **kwargs):
        return tf.concat(*args, **kwargs)

    def reduce_any(self, *args, **kwargs):
        return tf.math.reduce_any(*args, **kwargs).numpy()

    def where(self, *args, **kwargs):
        return tf.where(*args, **kwargs)

    def diag_part(self, *args, **kwargs):
        return tf.linalg.diag_part(*args, **kwargs)

    def diag(self, *args, **kwargs):
        return tf.linalg.diag(*args, **kwargs)

    def diag_operator(self, *args, **kwargs):
        return tf.linalg.LinearOperatorDiag(*args, **kwargs)

    def lu(self, x, *args, **kwargs):
        return tf.linalg.lu(x)

    def variable(self, x, *args, **kwargs):
        return tf.Variable(x, *args, **kwargs)

    def assign_cov(self, values):
        self.cov.assign(tf.constant(values))

    def maketensor(self, h5dset):
        if "original_shape" in h5dset.attrs:
            shape = h5dset.attrs["original_shape"]
        else:
            shape = h5dset.shape

        if h5dset.size == 0:
            return self.zeros(shape, h5dset.dtype)

        # read directly from hdf5 dataset to the underlying buffer of a tensor
        # this requires that the tensor is located on the CPU, so force the device
        with tf.device(tf.config.list_logical_devices("CPU")[0]):
            atensor = tf.zeros(h5dset.shape, h5dset.dtype)
            # zero tensors have a special flag set, using the identity clears this implicitly
            atensor = tf.identity(atensor)

        # read into the underlying array
        h5dset.read_direct(atensor.__array__())

        # the reshape operation is needed in case the hdf5 dataset was flattened
        # this also triggers a copy of the tensor to the default device (e.g. GPU)
        # if needed (ie if not the default CPU device)
        atensor = self.reshape(atensor, shape)
        return atensor

    def makesparsetensor(self, h5group):
        indices = self.maketensor(h5group["indices"])
        values = self.maketensor(h5group["values"])
        dense_shape = h5group.attrs["dense_shape"]

        return tf.sparse.SparseTensor(indices, values, dense_shape)
