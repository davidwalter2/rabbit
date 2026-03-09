
import hist
import tensorflow as tf

from rabbit.mappings import helpers
from rabbit.mappings.mapping import Mapping

import pdb

class HLT(Mapping):
    """
    A class to compute ratios of channels, processes, or bins.
    Optionally the numerator and denominator can be normalized.

    Parameters
    ----------
        indata: Input data used for analysis (e.g., histograms or data structures).
        h3_channel: str
            Name of the numerator channel.
        h2_channel: str
            Name of the denominator channel.
        h3_processes: list of str, optional
            List of process names for the numerator channel. Defaults to None, meaning all processes will be considered.
            Selected processes are summed before the ratio is computed.
        h2_processes: list of str, optional
            Same as h3_processes but for denumerator
        h3_selection: dict, optional
            Dictionary specifying selection criteria for the numerator. Keys are axis names, and values are slices or conditions.
            Defaults to an empty dictionary meaning no selection.
            E.g. {"charge":0, "ai":slice(0,2)}
            Selected axes are summed before the ratio is computed. To integrate over one axis before the ratio, use `slice(None)`
        h2_selection: dict, optional
            Same as h3_selection but for denumerator
        normalize: bool, optional
            Whether to normalize the numerator and denominator before the ratio. Defaults to False.
    """

    def __init__(
        self,
        indata,
        key,
        h3_channel,
        h2_channel,
        h1_channel,
        h3_processes=[],
        h2_processes=[],
        h1_processes = [],
        h3_selection={},
        h2_selection={},
        h1_selection={},
        h3_axes_rebin=[],
        h2_axes_rebin=[],
        h1_axes_rebin = [],
        h3_axes_sum=[],
        h2_axes_sum=[],
        h1_axes_sum= [],
    ):
        self.key = key

        self.h3 = helpers.Term(
            indata,
            h3_channel,
            h3_processes,
            h3_selection,
            h3_axes_rebin,
            h3_axes_sum,
        )
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

        self.has_data = self.h3.has_data and self.h2.has_data and self.h1.has_data

        self.need_processes = len(h3_processes) or len(
            h2_processes
        )  # the fun_flat will be by processes

        # The output of ratios will always be without process axis
        self.skip_per_process = True

        hist_axes = self.h2.channel_axes

        if h3_channel == h2_channel:
            channel = h3_channel
            flow = indata.channel_info[channel].get("flow", False)
        else:
            channel = f"{h3_channel}_{h2_channel}"
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
            <proc_h3_0>,<proc_h3_1>,... <proc_h3_0>,<proc_h3_1>,...
            <axis_h3_0>:<slice_h3_0>,<axis_h3_1>,<slice_h3_1>... <axis_h2_0>,<slice_h2_0>,<axis_h2_1>,<slice_h2_1>...

        Processes selections are optional. But in case on is given for the numerator, the denominator must be specified as well and vice versa.
        Use 'None' if you don't want to select any for either numerator xor denominator.

        Axes selections are optional. But in case one is given for the numerator, the denominator must be specified as well and vice versa.
        Use 'None:None' if you don't want to do any for either numerator xor denominator.
        """
        if len(args) > 3 and ":" not in args[2]:
            procs_h3 = [p for p in args[2].split(",") if p != "None"]
            procs_h2 = [p for p in args[3].split(",") if p != "None"]
            procs_h1 = [p for p in args[4].split(",") if p != "None"]

        else:
            procs_h3 = []
            procs_h2 = []
            procs_h1 = []

        # find axis selections
        if any(a for a in args if ":" in a):
            sel_args = [a for a in args if ":" in a]
        else:
            sel_args = ["None:None", "None:None", "None:None"]

        axis_selection_h3, axes_rebin_h3, axes_sum_h3 = helpers.parse_axis_selection(
            sel_args[0]
        )
        axis_selection_h2, axes_rebin_h2, axes_sum_h2 = helpers.parse_axis_selection(
            sel_args[1]
        )

        axis_selection_h1, axes_rebin_h1, axes_sum_h1 = helpers.parse_axis_selection(
            sel_args[2]
        )
        
        key = " ".join([cls.__name__, *args])


        return cls(
            indata,
            key,
            args[0],
            args[1],
            args[2],
            procs_h3,
            procs_h2,
            procs_h1,
            axis_selection_h3,
            axis_selection_h2,
            axis_selection_h1,
            axes_rebin_h3,
            axes_rebin_h2,
            axes_rebin_h1,
            axes_sum_h3,
            axes_sum_h2,
            axes_sum_h1,
        )

    def compute_flat(self, params, observables):
        h3 = self.h3.select(observables, inclusive=True)
        h2 = self.h2.select(observables, inclusive=True)
        h1 = self.h1.select(observables, inclusive=True)
        # original_shape = [24, 2, 10, 6] 
        # hlt_shape = [24, 2, 9, 6]
        
        # h3 = tf.reshape(h3, original_shape)
        # h2 = tf.reshape(h2, hlt_shape) ## need to make sure this reshaping is in the correct order
        h2_iso = h2[:, :1, :]
        h1_hlt = h1[:, 1:, :]
        # tf.reshape(h1, original_shape)
        
        h2_iso = tf.concat([h2_iso, h2], axis = 1)
        eps_iso = 2*h3/(h2_iso + 2*h3)
                
        ones = h2/h2
        eps_hlt = h2/(h2 + h1_hlt*(ones-eps_iso[:, 1:, :]))
        eps_hlt = tf.reshape(eps_hlt, [-1])

        return eps_hlt

    def compute_flat_per_process(self, params, observables):
        return self.compute_flat(params, observables)


class NormHLT(HLT):
    """
    Same as Ratio but the numerator and denominator are normalized
    """

    ndf_reduction = 1  

    def init(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_flat(self, params, observables):
        h3 = self.h3.select(observables, normalize = True, inclusive=True)
        h2 = self.h2.select(observables, normalize = True, inclusive=True)
        h1 = self.h1.select(observables, normalize = True, inclusive=True)
        original_shape = [24, 2, 10, 6] 
        hlt_shape = [24, 2, 9, 6]
        
        h3 = tf.reshape(h3, original_shape)
        h2 = tf.reshape(h2, hlt_shape) ## need to make sure this reshaping is in the correct order
        h2_hlt = h2[:, :, :1, :]
        h1_hlt = tf.reshape(h1, original_shape)[:, :, 1:, :]
        
        h2_iso = tf.concat([h2_hlt, h2], axis = 2)
        eps_iso = 2*h3/(h2_iso + 2*h3)
                
        ones = h2/h2
        eps_hlt = h2/(h2 + h1_hlt*(ones-eps_iso[:, :, 1:, :]))
        eps_hlt = tf.reshape(eps_hlt, [-1])

        return eps_hlt