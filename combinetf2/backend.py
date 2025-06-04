from wums import logging

logger = logging.child_logger(__name__)


class Backend:
    _shared_funcs = [
        "ones",
        "ones_like",
        "zeros",
        "zeros_like",
        "eye",
        "reshape",
        "where",
        "sqrt",
        "square",
        "squeeze",
        "concat",
        "matmul",
        "expand_dims",
    ]

    def __init__(self, options):
        self.bn = None

    def _initialize_shared_funcs(self):
        for name in self._shared_funcs:
            setattr(self, name, getattr(self.bn, name))

    def makesparsetensor(self, h5group):
        indices = self.maketensor(h5group["indices"])
        values = self.maketensor(h5group["values"])
        dense_shape = h5group.attrs["dense_shape"]

        return self.sparse.SparseTensor(indices, values, dense_shape)
