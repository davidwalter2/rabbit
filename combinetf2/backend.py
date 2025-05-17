from wums import logging

logger = logging.child_logger(__name__)


class Backend:
    def __init__(self, options):
        pass

    def makesparsetensor(self, h5group):
        indices = self.maketensor(h5group["indices"])
        values = self.maketensor(h5group["values"])
        dense_shape = h5group.attrs["dense_shape"]

        return self.sparse.SparseTensor(indices, values, dense_shape)
