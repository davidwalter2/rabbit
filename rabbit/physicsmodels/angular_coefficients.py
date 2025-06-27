import tensorflow as tf

from rabbit.physicsmodels import helpers
from rabbit.physicsmodels.physicsmodel import PhysicsModel


class AngularCoefficients(PhysicsModel):
    """
    A class to compute the angular coefficients as A_i = sigma_i / sigma_UL, result is a flat array of A_i and sigma_UL.

    Parameters
    ----------
        indata: Input data used for analysis (e.g., histograms or data structures).
        channel: str
            Name of the channel.
        process: str
            Name of the process.
        selection: dict, optional
            Dictionary specifying selection criteria. Keys are axis names, and values are slices or conditions.
            Defaults to an empty dictionary meaning no selection.
            E.g. {"charge":0, "ptVgen":slice(0,2), "absYVgen": hist.sum, "massVgen": hist.rebin(2)}
            Selected axes are summed before the ratio is computed. To integrate over one axis before the ratio, use `slice(None)`
    """

    def __init__(
        self,
        indata,
        key,
        channel,
        processes=[],
        selection={},
        rebin_axes={},
        sum_axes=[],
        selection_ul={},
        rebin_axes_ul={},
        sum_axes_ul=[],
        helicity_axis="helicitySig",
    ):
        self.key = key

        # sigma_i
        self.num = helpers.Term(
            indata,
            channel,
            processes,
            {helicity_axis: slice(1, None), **selection},
            rebin_axes,
            sum_axes,
        )

        # sigma_UL for Ais
        self.den = helpers.Term(
            indata,
            channel,
            processes,
            {helicity_axis: 0, **selection},
            rebin_axes,
            sum_axes,
        )

        # sigma_UL to keep in different binning
        self.sigma_ul = helpers.Term(
            indata,
            channel,
            processes,
            {helicity_axis: 0, **selection_ul},
            rebin_axes_ul,
            sum_axes_ul,
        )

        self.has_data = False

        self.need_processes = False

        # The output of ratios will always be without process axis
        self.skip_per_process = True

        self.has_processes = False  # The result has no process axis

        self.helicity_index = [
            i for i, a in enumerate(self.num.channel_axes) if a.name == helicity_axis
        ][0]

        channel_axes = []
        for a in self.num.channel_axes:
            if a.name == helicity_axis:
                a.__dict__["name"] = "ai"
            channel_axes.append(a)

        self.channel_info = {
            channel: {
                "axes": channel_axes,
            },
            f"{channel}_sigmaUL": {
                "axes": self.sigma_ul.channel_axes,
            },
        }

    @classmethod
    def parse_args(cls, indata, *args):
        """
        parsing the input arguments into the AI constructor, it has to be called as
        -m AngularCoefficients
            <ch>
            <proc_0>,<proc_1>,...
            <axis_ai_0>:<slice_ai_0>,<axis_ai_1>,<slice_ai_1>... <axis_sigma_UL_0>:<slice_sigma_UL_0>,<axis_sigma_UL_1>:<slice_sigma_UL_1>

        Processes selections are optional. Use 'None' if you don't want to select any.
        Axes selections are optional.
        """

        if len(args) > 2 and ":" not in args[1]:
            procs = [p for p in args[1].split(",") if p != "None"]
        else:
            procs = []

        # find axis selections
        axis_selection = {}
        axes_sum = []
        axes_rebin = {}

        axis_selection_ul = {}
        axes_sum_ul = []
        axes_rebin_ul = {}

        sel_args = [a for a in args if ":" in a]
        if len(sel_args):
            axis_selection, axes_rebin, axes_sum = helpers.parse_axis_selection(
                sel_args[0]
            )
        if len(sel_args) > 1:
            axis_selection_ul, axes_rebin_ul, axes_sum_ul = (
                helpers.parse_axis_selection(sel_args[1])
            )

        key = " ".join([cls.__name__, *args])

        return cls(
            indata,
            key,
            args[0],
            procs,
            axis_selection,
            axes_rebin,
            axes_sum,
            axis_selection_ul,
            axes_rebin_ul,
            axes_sum_ul,
        )

    def compute_ais(self, observables, inclusive=False):
        num = self.num.select(observables, inclusive=inclusive)
        den = self.den.select(observables, inclusive=inclusive)

        den = tf.expand_dims(den, axis=self.helicity_index)

        ai = num / den
        ai_flat = tf.reshape(ai, [-1])

        sigma_ul = self.sigma_ul.select(observables, inclusive=inclusive)
        sigma_ul_flat = tf.reshape(sigma_ul, [-1])

        return tf.concat([ai_flat, sigma_ul_flat], axis=0)

    def compute_flat(self, params, observables):
        return self.compute_ais(observables, True)

    def compute_flat_per_process(self, params, observables):
        return self.compute_ais(observables, False)


class LamTung(AngularCoefficients):

    def __init__(self, indata, key, channel, *args, **kwargs):
        super().__init__(indata, key, channel, *args, **kwargs)

        self.channel_info[channel]["axes"] = [
            c for c in self.channel_info[channel]["axes"] if c.name != "ai"
        ]

    def compute_ais(self, observables, inclusive=False):
        ais = super().compute_ais(observables, inclusive)
        a0 = tf.gather(ais, indices=0, axis=self.helicity_index)
        a2 = tf.gather(ais, indices=2, axis=self.helicity_index)
        return a0 - a2
