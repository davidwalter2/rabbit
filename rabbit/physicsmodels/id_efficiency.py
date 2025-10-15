import hist
import tensorflow as tf

from rabbit.physicsmodels import helpers
from rabbit.physicsmodels.physicsmodel import PhysicsModel

import pdb

class ID(PhysicsModel):
    """
    A class to compute ratios of channels, processes, or bins.
    Optionally the numerator and denominator can be normalized.

    Parameters
    ----------
        indata: Input data used for analysis (e.g., histograms or data structures).
        h2_channel: str
            Name of the numerator channel.
        h1_channel: str
            Name of the denominator channel.
        h2_processes: list of str, optional
            List of process names for the numerator channel. Defaults to None, meaning all processes will be considered.
            Selected processes are summed before the ratio is computed.
        h1_processes: list of str, optional
            Same as h2_processes but for denumerator
        h2_selection: dict, optional
            Dictionary specifying selection criteria for the numerator. Keys are axis names, and values are slices or conditions.
            Defaults to an empty dictionary meaning no selection.
            E.g. {"charge":0, "ai":slice(0,2)}
            Selected axes are summed before the ratio is computed. To integrate over one axis before the ratio, use `slice(None)`
        h1_selection: dict, optional
            Same as h2_selection but for denumerator
        normalize: bool, optional
            Whether to normalize the numerator and denominator before the ratio. Defaults to False.
    """

    def __init__(
        self,
        indata,
        key,
        h2_channel,
        h1_channel,
        h0_channel,
        h2_processes=[],
        h1_processes=[],
        h0_processes = [],
        h2_selection={},
        h1_selection={},
        h0_selection={},
        h2_axes_rebin=[],
        h1_axes_rebin=[],
        h0_axes_rebin = [],
        h2_axes_sum=[],
        h1_axes_sum=[],
        h0_axes_sum= [],
    ):
        self.key = key

        self.h2 = helpers.Term(
            indata,
            h2_channel,
            h2_processes,
            h2_selection,
            h2_axes_rebin,
            h2_axes_sum,
        )
        self.h1 = helpers.Term(
            indata,
            h1_channel,
            h1_processes,
            h1_selection,
            h1_axes_rebin,
            h1_axes_sum,
        )
        
        self.h0 = helpers.Term(
            indata,
            h0_channel,
            h0_processes,
            h0_selection,
            h0_axes_rebin,
            h0_axes_sum,
        )

        self.has_data = self.h2.has_data and self.h1.has_data and self.h0.has_data

        self.need_processes = len(h2_processes) or len(
            h1_processes
        )  # the fun_flat will be by processes

        # The output of ratios will always be without process axis
        self.skip_per_process = True

        # if [a.size for a in self.h2.channel_axes] != [
        #     a.size for a in self.h1.channel_axes
        # ]:
        #     raise RuntimeError(
        #         "Channel axes for numerator and denominator must have the same number of bins"
        #     )
        
        hist_axes = self.h0.channel_axes

        if h2_channel == h1_channel:
            channel = h2_channel
            flow = indata.channel_info[channel].get("flow", False)
        else:
            channel = f"{h2_channel}_{h1_channel}"
            flow = False

        self.has_processes = False  # The result has no process axis

        self.channel_info = {
            channel: {
                "axes": hist_axes,
                "flow": flow,
            }
        }

    @classmethod
    def parse_args(cls, indata, *args):
        """
        parsing the input arguments into the ratio constructor, is has to be called as
        -m Ratio
            <ch num> <ch den>
            <proc_h2_0>,<proc_h2_1>,... <proc_h2_0>,<proc_h2_1>,...
            <axis_h2_0>:<slice_h2_0>,<axis_h2_1>,<slice_h2_1>... <axis_h1_0>,<slice_h1_0>,<axis_h1_1>,<slice_h1_1>...

        Processes selections are optional. But in case on is given for the numerator, the denominator must be specified as well and vice versa.
        Use 'None' if you don't want to select any for either numerator xor denominator.

        Axes selections are optional. But in case one is given for the numerator, the denominator must be specified as well and vice versa.
        Use 'None:None' if you don't want to do any for either numerator xor denominator.
        """
        if len(args) > 3 and ":" not in args[2]:
            procs_h2 = [p for p in args[2].split(",") if p != "None"]
            procs_h1 = [p for p in args[3].split(",") if p != "None"]
            procs_h0 = [p for p in args[4].split(",") if p != "None"]

        else:
            procs_h2 = []
            procs_h1 = []
            procs_h0 = []

        # find axis selections
        if any(a for a in args if ":" in a):
            sel_args = [a for a in args if ":" in a]
        else:
            sel_args = ["None:None", "None:None", "None:None"]

        axis_selection_h2, axes_rebin_h2, axes_sum_h2 = helpers.parse_axis_selection(
            sel_args[0]
        )
        axis_selection_h1, axes_rebin_h1, axes_sum_h1 = helpers.parse_axis_selection(
            sel_args[1]
        )

        axis_selection_h0, axes_rebin_h0, axes_sum_h0 = helpers.parse_axis_selection(
            sel_args[2]
        )
        
        key = " ".join([cls.__name__, *args])


        return cls(
            indata,
            key,
            args[0],
            args[1],
            args[2],
            procs_h2,
            procs_h1,
            procs_h0,
            axis_selection_h2,
            axis_selection_h1,
            axis_selection_h0,
            axes_rebin_h2,
            axes_rebin_h1,
            axes_rebin_h0,
            axes_sum_h2,
            axes_sum_h1,
            axes_sum_h0,
        )

    def compute_flat(self, params, observables):
        h2 = self.h2.select(observables, inclusive=True)
        h1_hlt = self.h1.select(observables, inclusive=True)[:, 1:, :]
        h1 = self.h1.select(observables, inclusive=True)
        h0 = self.h0.select(observables, inclusive=True)


        eps_hlt = 2*h2/(h1_hlt + 2*h2)
        
        eps_hlt_expanded = tf.zeros(shape=[2, 1, 3], dtype = tf.float64)  # or tf.ones, or any values you want
        eps_hlt_expanded = tf.concat([eps_hlt_expanded, eps_hlt], axis = 1)
       
        eps_id = h1/(h1 + h0*(1-eps_hlt_expanded))
        # pdb.set_trace()
        eps_id = tf.reshape(eps_id, [-1])

        return eps_id

    def compute_flat_per_process(self, params, observables):
        return self.compute_flat(params, observables)


class NormID(ID):
    """
    Same as Ratio but the numerator and denominator are normalized
    """

    ndf_reduction = 1

    def init(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_flat(self, params, observables):
        h2 = self.h2.select(observables, normalize = True, inclusive=True)
        h1_hlt = self.h1.select(observables,normalize = True,  inclusive=True)[:, 1:, :]
        h1 = self.h1.select(observables, normalize = True, inclusive=True)
        h0 = self.h0.select(observables, normalize = True, inclusive=True)


        eps_hlt = 2*h2/(h1_hlt + 2*h2)
        
        eps_hlt_expanded = tf.zeros(shape=[2, 1, 3], dtype = tf.float64)  # or tf.ones, or any values you want
        eps_hlt_expanded = tf.concat([eps_hlt_expanded, eps_hlt], axis = 1)
       
        eps_id = h1/(h1 + h0*(1-eps_hlt_expanded))
        # pdb.set_trace()
        eps_id = tf.reshape(eps_id, [-1])

        return eps_id