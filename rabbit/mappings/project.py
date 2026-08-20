import tensorflow as tf

from rabbit.mappings.mapping import Select


class Project(Select):
    """
    A class to project a histogram to lower dimensions, i.e. all axes of the channel
    that are not kept are summed. The output axes are in the order they are requested.

    Parameters
    ----------
    channel : str
        Name of the channel. Required.
    axes_names : list of str, optional
        Names of the axes to keep. If empty, the histogram will be projected to a single bin.
    """

    def __init__(self, indata, key, channel, *axes_names, **kwargs):
        channel_axes_names = [a.name for a in indata.channel_info[channel]["axes"]]

        missing = [n for n in axes_names if n not in channel_axes_names]
        if len(missing):
            raise RuntimeError(
                f"Axes {missing} not found. Available axes are {channel_axes_names}"
            )

        # a projection is a selection of the channel where all axes that are not kept are summed
        super().__init__(
            indata,
            key,
            channel,
            sum_axes=[n for n in channel_axes_names if n not in axes_names],
            **kwargs,
        )

        # the axes that are kept are in the order of the channel, the result is transposed
        #   into the order in which the axes were requested
        kept_axes_names = [n for n in channel_axes_names if n in axes_names]
        self.transpose_idxs = [kept_axes_names.index(n) for n in axes_names]

        info = self.channel_info[channel]
        info["axes"] = [info["axes"][i] for i in self.transpose_idxs]

    @classmethod
    def parse_args(cls, indata, channel, *axes_names):
        """
        parsing the input arguments into the constructor, it has to be called as
        -m Project <ch> <axis_0> <axis_1> ...

        All axes of the channel that are not listed are summed,
        listing no axis at all gives the total yield.
        """
        key = " ".join([cls.__name__, channel, *axes_names])

        return cls(indata, key, channel, *axes_names)

    def compute(self, params, observables):
        if self.transpose_idxs == sorted(self.transpose_idxs):
            # axes are already in the requested order
            return observables

        perm = self.transpose_idxs[:]
        if len(observables.shape) > len(perm):
            # last is process axis
            perm += list(range(len(perm), len(observables.shape)))

        return tf.transpose(observables, perm=perm)


class Normalize(Project):
    """
    Same as Project but the result is normalized to its integral, summed over all processes
    """

    ndf_reduction = 1
    normalize = True
