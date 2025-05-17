from wums import logging

from combinetf2.fitter import Fitter

logger = logging.child_logger(__name__)


class FitterJax(Fitter):
    def __init__(self, bn, indata, options):
        super().__init__(bn, indata, options)

        # tf.config.experimental.enable_op_determinism()

        if options.eager:
            # TODO
            pass

        # tf.random.set_seed(options.seed)
