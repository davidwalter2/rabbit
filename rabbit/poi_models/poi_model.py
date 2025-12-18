import tensorflow as tf


class POIModel:

    def __init__(self, indata, **kwargs):
        self.indata = indata
        self.npoi = 0
        self.pois = []
        self.poidefault = tf.zeros([], dtype=self.indata.dtype)

        self.allowNegativePOI = False

        self.is_linear = True  # determines of a model is linear or not

    def compute(self, poi):
        """
        Compute an array for the rate per process
        :param params: 1D tensor of explicit parameters in the fit
        """

        rnorm = tf.ones(self.indata.nproc, dtype=self.indata.dtype)

        return rnorm


class Mu(POIModel):

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
        """
        Compute an array for the rate per process
        :param params: 1D tensor of explicit parameters in the fit
        """

        rnorm = tf.concat(
            [poi, tf.ones([self.indata.nproc - poi.shape[0]], dtype=self.indata.dtype)],
            axis=0,
        )

        return rnorm
