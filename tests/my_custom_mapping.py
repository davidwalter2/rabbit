# write the following to a file
import tensorflow as tf

from rabbit.mappings import helpers
from rabbit.mappings.ratio import Ratio


class Custom(Ratio):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_flat(self, params, observables):
        x = self.num.select(observables, inclusive=True)
        y = self.den.select(observables, inclusive=True)

        r = tf.reshape(x / y, [-1])

        result = x * tf.square(tf.sin(r))
        return result
