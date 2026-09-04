# write the following to a file
import hist
import numpy as np
import tensorflow as tf

from rabbit.param_models.param_model import ParamModel


class Custom(ParamModel):

    def __init__(self, indata, expectSignal=None, allowNegativeParam=False, **kwargs):
        self.indata = indata

        self.npoi = self.indata.nsignals
        self.npou = 0

        self.params = np.array([s for s in self.indata.signals])

        self.allowNegativeParam = allowNegativeParam

        self.is_linear = self.nparams == 0 or self.allowNegativeParam

        self.set_param_default(expectSignal, allowNegativeParam)

        # track which bins belong to cat:0 vs cat:1
        flat_values = []
        flat_values_masked = []
        for i, c in indata.channel_info.items():
            hi = hist.Hist(*c["axes"])

            if len(hi.axes) == 1:
                hi.values()[...] = np.array([0, 1])
            else:
                hi[{"cat": 0}].values()[...] = 0
                hi[{"cat": 1}].values()[...] = 1

            if c["masked"]:
                flat_values_masked.append(hi.values().flatten().astype("float64"))
            else:
                flat_values.append(hi.values().flatten().astype("float64"))

        self.is_cat1 = tf.concat(flat_values, axis=0)
        self.is_cat1_full = tf.concat([*flat_values, *flat_values_masked], axis=0)

    def compute(self, param, full=False):
        if full:
            nbins = self.indata.nbinsfull
            is_cat1 = self.is_cat1_full
        else:
            nbins = self.indata.nbins
            is_cat1 = self.is_cat1

        signal_slice = (1 - is_cat1) * tf.square(tf.sin(param)) + is_cat1 * tf.square(
            tf.cos(param)
        )
        signal_slice = tf.reshape(signal_slice, [1, -1])
        bkg_slice = tf.ones(
            [self.indata.nproc - param.shape[0], nbins], dtype=self.indata.dtype
        )

        rnorm = tf.concat(
            [signal_slice, bkg_slice],
            axis=0,
        )

        rnorm = tf.transpose(rnorm)

        return rnorm
