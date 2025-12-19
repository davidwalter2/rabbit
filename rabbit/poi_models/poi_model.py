import numpy as np
import tensorflow as tf


class POIModel:

    def __init__(self, indata, **kwargs):
        self.indata = indata

        # a POI model must set these attribues
        # self.npoi = # number of parameters of interest (POIs)
        # self.pois = # list of names for the POIs
        # self.poidefault = # default values for the POIs
        # self.is_linear = # define if the model is linear in the POIs
        # self.allowNegativePOI = # define if the POI can be negative or not

    def compute(self, poi):
        """
        Compute an array for the rate per process
        :param params: 1D tensor of explicit parameters in the fit
        :return 2D tensor to be multiplied with [proc,bin] tensor
        """


class Ones(POIModel):
    """
    multiply all processes with ones
    """

    def __init__(self, indata, **kwargs):
        self.indata = indata
        self.npoi = 0
        self.pois = []
        self.poidefault = tf.zeros([], dtype=self.indata.dtype)

        self.allowNegativePOI = False

    def compute(self, poi):
        rnorm = tf.ones(self.indata.nproc, dtype=self.indata.dtype)
        rnorm = tf.reshape(rnorm, [1, -1])
        return rnorm


class Mu(POIModel):
    """
    multiply unconstrained parameter to signal processes, and ones otherwise
    """

    def __init__(self, indata, expectSignal=1, allowNegativePOI=False, **kwargs):
        self.indata = indata

        self.npoi = self.indata.nsignals

        self.pois = []
        for signal in self.indata.signals:
            self.pois.append(signal)

        self.allowNegativePOI = allowNegativePOI

        self.is_linear = self.npoi == 0 or self.allowNegativePOI

        poidefault = expectSignal * tf.ones([self.npoi], dtype=self.indata.dtype)
        if self.allowNegativePOI:
            self.xpoidefault = poidefault
        else:
            self.xpoidefault = tf.sqrt(poidefault)

    def compute(self, poi):
        rnorm = tf.concat(
            [poi, tf.ones([self.indata.nproc - poi.shape[0]], dtype=self.indata.dtype)],
            axis=0,
        )

        rnorm = tf.reshape(rnorm, [1, -1])
        return rnorm


class MixtureModel(POIModel):
    """
    Based on unconstrained parameters x_i
    multiply `primary` process by x_i
    multiply `complementary` process by 1-x_i
    """

    def __init__(
        self,
        indata,
        primary_processes="sig",
        complementary_processes="bkg",
        expectSignal=0,
        allowNegativePOI=False,
        **kwargs,
    ):
        self.indata = indata

        if type(primary_processes) == str:
            primary_processes = [primary_processes]

        if type(complementary_processes) == str:
            complementary_processes = [complementary_processes]

        primary_processes = np.array(primary_processes).astype("S")
        complementary_processes = np.array(complementary_processes).astype("S")

        if len(primary_processes) != len(complementary_processes):
            raise ValueError(
                f"Length of pimary and complementary processes has to be the same, but got {len(primary_processes)} and {len(complementary_processes)}"
            )

        self.primary_idxs = np.where(np.isin(self.indata.procs, primary_processes))[0]
        self.complementary_idxs = np.where(
            np.isin(self.indata.procs, complementary_processes)
        )[0]
        self.all_idx = np.concatenate([self.primary_idxs, self.complementary_idxs])

        self.npoi = len(primary_processes)
        self.pois = [
            f"{p}_{c}_mixing"
            for p, c in zip(primary_processes, complementary_processes)
        ]

        self.allowNegativePOI = allowNegativePOI
        self.is_linear = False

        poidefault = expectSignal * tf.ones([self.npoi], dtype=self.indata.dtype)
        if self.allowNegativePOI:
            self.xpoidefault = poidefault
        else:
            self.xpoidefault = tf.sqrt(poidefault)

    def compute(self, poi):

        all_updates = tf.concat(
            [
                tf.fill([len(self.primary_idxs)], poi),
                tf.fill([len(self.complementary_idxs)], 1 - poi),
            ],
            axis=0,
        )

        # Single scatter update
        rnorm = tf.tensor_scatter_nd_update(
            tf.ones(self.indata.nproc, dtype=self.indata.dtype),
            self.all_idx[:, None],
            all_updates,
        )

        rnorm = tf.reshape(rnorm, [1, -1])
        return rnorm
