#!/usr/bin/env python3

"""
Plot several channels stitched together along one continuous axis (by default mll),
e.g. ch_iso_poi_low (mll < 76) + ch_iso_eff (76 < mll < 106) + ch_iso_poi_high (mll > 106).

Channels without the stitch axis (like the efficiency channel at the Z peak) are
included as a single bin along the stitch axis, with edges taken from the neighbouring
channels or from --stitchEdges. All axes not listed in --keepAxes are summed.

The plotting itself is done by rabbit_plot_hists.py, so all of its options
(variations, --varFiles, --varGroups, ratios, ...) work the same way.
"""

import importlib.util
import os

import hist
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

import rabbit.io_tools

from wums import logging, output_tools, plot_tools  # isort: skip

# rabbit_plot_hists.py lives next to this script and is not part of the rabbit package
_spec = importlib.util.spec_from_file_location(
    "rabbit_plot_hists",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "rabbit_plot_hists.py"),
)
rph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rph)

logger = None


def make_parser():
    parser = rph.make_parser()
    parser.add_argument(
        "--stitchChannels",
        type=str,
        nargs="+",
        default=["ch_iso_poi_low", "ch_iso_eff", "ch_iso_poi_high"],
        help="Channels to stitch, in increasing order along the stitch axis",
    )
    parser.add_argument(
        "--stitchAxis",
        type=str,
        default="mll",
        help="Axis along which the channels are stitched",
    )
    parser.add_argument(
        "--stitchEdges",
        type=float,
        nargs="*",
        default=[],
        help="""
        Lower and upper edge (2 values per channel) along the stitch axis for each channel
        that does not have the stitch axis, in the order of --stitchChannels.
        By default the edges are taken from the neighbouring channels.
        """,
    )
    parser.add_argument(
        "--keepAxes",
        type=str,
        nargs="+",
        default=["mll"],
        help="""
        Axes to keep (all others are summed), must include the stitch axis.
        Combine with --selectionAxes to make one plot per bin of an additional axis,
        e.g. '--keepAxes time mll --selectionAxes time'
        """,
    )
    return parser


class LazyHist:
    """Mimics the '.get()' interface of the fit result entries, builds the hist on first use"""

    def __init__(self, func):
        self.func = func
        self.h = None

    def get(self):
        if self.h is None:
            self.h = self.func()
        return self.h


def find_source(mappings, channel, channel_axes, keep_axes):
    """
    Return the fit result for the channel with the kept axes and whether the
    uncertainties are exact. Uses a 'Project <channel> <axes>' mapping if it exists,
    otherwise the BaseMapping, where summing bins neglects bin-by-bin correlations.
    """
    kept = [n for n in keep_axes if n in channel_axes]
    base = mappings["BaseMapping"]["channels"][channel]
    if len(kept) == len(channel_axes):
        return base, True

    for key in mappings.keys():
        tokens = key.split(" ")
        if (
            tokens[0] == "Project"
            and len(tokens) > 1
            and tokens[1] == channel
            and sorted(tokens[2:]) == sorted(kept)
        ):
            logger.info(f"Use mapping '{key}' for channel {channel}")
            return mappings[key]["channels"][channel], True

    summed = [n for n in channel_axes if n not in kept]
    project = " ".join(["Project", channel, *kept])
    logger.warning(
        f"No mapping '{project}' found, sum axes {summed} of the BaseMapping instead. "
        f"Uncertainty bands and grouped impacts then neglect bin-by-bin correlations; "
        f"for exact uncertainties run rabbit_fit with '-m {project}'"
    )
    return base, False


def reduce_hist(h, channel_axes, keep_axes, stitch_axis):
    """
    Sum all channel axes that are not kept and order the axes as
    (kept axes, extra axes), where extra axes are e.g. processes or vars.
    Returns arrays with a (possibly length-1) stitch axis and the axis objects.
    """
    names = list(h.axes.name)
    kept = [n for n in keep_axes if n in names]
    extra = [n for n in names if n not in channel_axes]
    summed = [n for n in names if n not in kept and n not in extra]

    values = np.asarray(h.values(flow=False))
    variances = h.variances(flow=False)
    if variances is not None:
        variances = np.asarray(variances)

    def _reduce(arr):
        arr = arr.sum(axis=tuple(names.index(n) for n in summed))
        remaining = [n for n in names if n not in summed]
        arr = np.transpose(arr, [remaining.index(n) for n in kept + extra])
        if stitch_axis not in kept:
            arr = np.expand_dims(arr, keep_axes.index(stitch_axis))
        return arr

    values = _reduce(values)
    if variances is not None:
        variances = _reduce(variances)

    axes = {n: h.axes[n] for n in kept + extra}
    return values, variances, axes, extra


def stitch_hists(hists, channels, channel_axes, keep_axes, stitch_axis, edges):
    reduced = [
        reduce_hist(h, channel_axes[c], keep_axes, stitch_axis)
        for h, c in zip(hists, channels)
    ]

    extra = reduced[0][3]
    for (_, _, axes, ex), c in zip(reduced[1:], channels[1:]):
        if ex != extra:
            raise ValueError(f"Channel {c} has extra axes {ex}, expected {extra}")

    # all non stitched axes have to be identical
    out_axes = []
    for n in keep_axes + extra:
        if n == stitch_axis:
            out_axes.append(
                hist.axis.Variable(edges, name=n, underflow=False, overflow=False)
            )
            continue
        ax = reduced[0][2][n]
        for (_, _, axes, _), c in zip(reduced[1:], channels[1:]):
            if axes[n] != ax:
                raise ValueError(
                    f"Axis '{n}' of channel {c} differs from the one of {channels[0]}"
                )
        out_axes.append(ax)

    i_stitch = keep_axes.index(stitch_axis)
    values = np.concatenate([r[0] for r in reduced], axis=i_stitch)
    if all(r[1] is not None for r in reduced):
        variances = np.concatenate([r[1] for r in reduced], axis=i_stitch)
        h = hist.Hist(*out_axes, storage=hist.storage.Weight())
        h.values()[...] = values
        h.variances()[...] = variances
    else:
        h = hist.Hist(*out_axes, storage=hist.storage.Double())
        h.values()[...] = values
    return h


def stitch_edges(channels, channel_axes, stitch_axis, explicit_edges):
    """Bin edges along the stitch axis for each channel, in channel order"""
    missing = [c for c in channels if stitch_axis not in channel_axes[c]]
    if len(explicit_edges) not in [0, 2 * len(missing)]:
        raise ValueError(
            f"--stitchEdges needs 2 values for each of the channels {missing} without axis '{stitch_axis}'"
        )

    edges = {
        c: channel_axes[c][stitch_axis].edges
        for c in channels
        if stitch_axis in channel_axes[c]
    }
    for i, c in enumerate(missing):
        if len(explicit_edges):
            edges[c] = np.array(explicit_edges[2 * i : 2 * i + 2])
            continue
        j = channels.index(c)
        if (
            j == 0
            or j == len(channels) - 1
            or channels[j - 1] not in edges
            or channels[j + 1] not in edges
        ):
            raise ValueError(
                f"Can not infer the '{stitch_axis}' range of channel {c} from its neighbours, use --stitchEdges"
            )
        edges[c] = np.array([edges[channels[j - 1]][-1], edges[channels[j + 1]][0]])

    all_edges = [edges[channels[0]]]
    for prev, c in zip(channels[:-1], channels[1:]):
        if not np.isclose(edges[prev][-1], edges[c][0]):
            raise ValueError(
                f"Channels {prev} and {c} are not contiguous along '{stitch_axis}': {edges[prev][-1]} != {edges[c][0]}"
            )
        all_edges.append(edges[c][1:])
    return np.concatenate(all_edges)


def stitch_result(fitresult, meta, args):
    """Build a dict mimicking a mapping channel result with the stitched histograms"""
    mappings = fitresult.get("mappings", fitresult.get("physics_models"))
    channel_info = meta["meta_info_input"]["channel_info"]
    channels = args.stitchChannels

    for c in channels:
        if c not in channel_info:
            raise ValueError(
                f"Channel {c} not found, available channels are {list(channel_info.keys())}"
            )
    channel_axes = {
        c: {a.name: a for a in channel_info[c]["axes"]} for c in channels
    }

    edges = stitch_edges(channels, channel_axes, args.stitchAxis, args.stitchEdges)
    logger.info(f"Stitched '{args.stitchAxis}' edges: {edges}")

    sources = [
        find_source(mappings, c, list(channel_axes[c].keys()), args.keepAxes)[0]
        for c in channels
    ]
    keys = set.intersection(*[set(s.keys()) for s in sources])

    def _make(key):
        return lambda: stitch_hists(
            [s[key].get() for s in sources],
            channels,
            channel_axes,
            args.keepAxes,
            args.stitchAxis,
            edges,
        )

    return {k: LazyHist(_make(k)) for k in keys if k.startswith("hist_")}


def main():
    args = make_parser().parse_args()
    global logger
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)
    rph.logger = logger

    if args.stitchAxis not in args.keepAxes:
        args.keepAxes = [args.stitchAxis, *args.keepAxes]
    if args.dataCovariance:
        raise NotImplementedError("--dataCovariance is not supported for stitched plots")
    if args.unfoldedXsec:
        raise NotImplementedError("--unfoldedXsec is not supported for stitched plots")

    config = plot_tools.load_config(args.config)

    varNames = args.varNames
    variation_names = []
    if varNames is not None:
        variation_names.extend(varNames)
    variation_names.extend(args.varGroups)

    varLabels = args.varLabels
    varColors = args.varColors
    if variation_names:
        if varLabels is None:
            syst_labels = getattr(config, "systematics_labels", {})
            varLabels = [syst_labels.get(x, x) for x in variation_names]
        elif len(varLabels) != len(variation_names):
            raise ValueError(
                f"Must specify the same number of args for variation names and --varLabels. Found variations={len(variation_names)} and varLabels={len(varLabels)}"
            )
        if varColors is None:
            varColors = [
                colormaps["tab10" if len(variation_names) < 10 else "tab20"](i)
                for i in range(len(variation_names))
            ]

    if len(args.fillVariationsAlphas) == 1 and len(variation_names) > 1:
        args.fillVariationsAlphas = args.fillVariationsAlphas * len(variation_names)
    elif len(args.fillVariationsAlphas) not in [0, len(variation_names)]:
        raise ValueError(
            f"Must specify either zero, one, or exactly one alpha per variation with --fillVariationsAlphas. Found {len(args.fillVariationsAlphas)} alphas for {len(variation_names)} variations."
        )

    fittype = "prefit" if args.prefit else "postfit"

    fitresult, meta = rabbit.io_tools.get_fitresult(args.infile, args.result, meta=True)
    result = stitch_result(fitresult, meta, args)

    varResults = []
    for f in args.varFiles:
        r, m = rabbit.io_tools.get_fitresult(f, args.result, meta=True)
        varResults.append(stitch_result(r, m, args))

    plt.rcParams["font.size"] = plt.rcParams["font.size"] * args.scaleTextSize

    channel_info = meta["meta_info_input"]["channel_info"]

    procs = meta["procs"].astype(str)[::-1]
    if args.filterProcs is not None:
        procs = [p for p in procs if p in args.filterProcs]

    if hasattr(config, "get_labels_colors_procs_sorted"):
        labels, colors, procs = config.get_labels_colors_procs_sorted(procs)
    else:
        labels = procs[:]
        cmap = plt.get_cmap("tab10")
        proc_colors = getattr(config, "process_colors", {})
        colors = [proc_colors.get(p, cmap(i % cmap.N)) for i, p in enumerate(procs)]

    if args.lumi is None:
        lumi = [c["lumi"] for c in channel_info.values() if "lumi" in c.keys()]
        lumi = sorted(set(lumi))
        lumi = lumi[0] if len(lumi) == 1 else (None if len(lumi) == 0 else lumi)
    else:
        lumi = args.lumi

    outdir = output_tools.make_plot_dir(args.outpath, eoscp=args.eoscp)

    rph.make_plots(
        result,
        outdir,
        config,
        procs=procs,
        labels=labels,
        colors=colors,
        args=args,
        channel="stitched",
        lumi=lumi,
        fittype=fittype,
        varResults=varResults,
        varFilesFitTypes=args.varFilesFitTypes,
        varMarkers=args.varMarkers,
        varNames=varNames,
        varLabels=varLabels,
        varColors=varColors,
        binwnorm=None if args.noBinWidthNorm else 1.0,
        chi2=[None, None],
        saturated_chi2=False,
        meta=meta,
        is_normalized=False,
        counts=True,
        dataCovariance=False,
    )

    if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
        output_tools.copy_to_eos(outdir, args.outpath)


if __name__ == "__main__":
    main()
