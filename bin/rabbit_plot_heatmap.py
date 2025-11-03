#!/usr/bin/env python3

import argparse
import inspect
import itertools
import os

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import colormaps
from matplotlib.lines import Line2D


import seaborn as sns
import rabbit.io_tools

from wums import boostHistHelpers as hh  # isort: skip
from wums import logging, output_tools, plot_tools  # isort: skip

import pdb

hep.style.use(hep.style.ROOT)

logger = None


def parseArgs():

    # choices for legend padding
    choices_padding = ["auto", "lower left", "lower right", "upper left", "upper right"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4],
        help="Set verbosity level with logging, the larger the more verbose",
    )
    parser.add_argument(
        "--noColorLogger", action="store_true", help="Do not use logging with colors"
    )
    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        default=os.path.expanduser("./test"),
        help="Base path for output",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file for style formatting",
    )
    parser.add_argument(
        "--eoscp",
        action="store_true",
        help="Override use of xrdcp and use the mount instead",
    )
    parser.add_argument(
        "-p", "--postfix", type=str, help="Postfix for output file name"
    )
    parser.add_argument(
        "--lumi",
        type=float,
        default=16.8,
        help="Luminosity used in the fit, needed to get the absolute cross section",
    )
    parser.add_argument(
        "--title",
        default="Rabbit",
        type=str,
        help="Title to be printed in upper left",
    )
    parser.add_argument(
        "--subtitle",
        default="",
        type=str,
        help="Subtitle to be printed after title",
    )
    parser.add_argument("--titlePos", type=int, default=2, help="title position")
    parser.add_argument(
        "--legPos", type=str, default="upper right", help="Set legend position"
    )
    parser.add_argument(
        "--legSize",
        type=str,
        default="small",
        help="Legend text size (small: axis ticks size, large: axis label size, number)",
    )
    parser.add_argument(
        "--legCols", type=int, default=2, help="Number of columns in legend"
    )
    parser.add_argument(
        "--legPadding",
        type=str,
        default="auto",
        choices=choices_padding,
        help="Where to put empty entries in legend",
    )
    parser.add_argument(
        "--lowerLegPos",
        type=str,
        default="upper left",
        help="Set lower legend position",
    )
    parser.add_argument(
        "--lowerLegCols", type=int, default=2, help="Number of columns in lower legend"
    )
    parser.add_argument(
        "--lowerLegPadding",
        type=str,
        default="auto",
        choices=choices_padding,
        help="Where to put empty entries in lower legend",
    )
    parser.add_argument(
        "--noSciy",
        action="store_true",
        help="Don't allow scientific notation for y axis",
    )
    parser.add_argument(
        "--yscale",
        type=float,
        help="Scale the upper y axis by this factor (useful when auto scaling cuts off legend)",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        help="Min and max values for y axis (if not specified, range set automatically)",
    )
    parser.add_argument("--xlim", type=float, nargs=2, help="min and max for x axis")
    parser.add_argument(
        "--rrange",
        type=float,
        nargs=2,
        default=[0.9, 1.1],
        help="y range for ratio plot",
    )
    parser.add_argument(
        "--scaleTextSize",
        type=float,
        default=1.0,
        help="Scale all text sizes by this number",
    )
    parser.add_argument(
        "--customFigureWidth",
        type=float,
        default=None,
        help="Use a custom figure width, otherwise chosen automatic",
    )
    parser.add_argument(
        "infile",
        type=str,
        help="hdf5 file from rabbit or root file from combinetf",
    )
    parser.add_argument(
        "--result",
        default=None,
        type=str,
        help="fitresults key in file (e.g. 'asimov'). Leave empty for data fit result.",
    )
    parser.add_argument(
        "--logy", action="store_true", help="Make the yscale logarithmic"
    )
    parser.add_argument(
        "--noLowerPanel",
        action="store_true",
        help="Don't plot the lower panel in the plot",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Plot difference in lower panel instead of ratio",
    )
    parser.add_argument(
        "--dataHist",
        type=str,
        default="nobs",
        choices=["data_obs", "nobs"],
        help="Which data to plot ('data_obs': data histogram provided in input data; 'nobs': Plot (pseudo) data used in the fit)",
    )
    parser.add_argument("--noData", action="store_true", help="Don't plot the data")
    parser.add_argument(
        "--noUncertainty", action="store_true", help="Don't plot total uncertainty band"
    )
    parser.add_argument(
        "--normToData", action="store_true", help="Normalize MC to data"
    )
    parser.add_argument(
        "--prefit", action="store_true", help="Make prefit plot, else postfit"
    )
    parser.add_argument(
        "-m",
        "--physicsModel",
        nargs="+",
        action="append",
        default=[],
        help="""
        Make plot of physics model prefit and postfit histograms. Loop over all by deault. 
        Can also specify the model name, followed by the arguments, e.g. "-m Project ch0 eta pt". 
        This argument can be called multiple times.
        """,
    )
    parser.add_argument(
        "--filterProcs",
        type=str,
        nargs="*",
        default=None,
        help="Only plot the filtered processes",
    )
    parser.add_argument(
        "--suppressProcsLabel",
        type=str,
        nargs="*",
        default=[],
        help="Don't show given processes in the legends",
    )
    parser.add_argument(
        "--channels",
        type=str,
        nargs="*",
        default=None,
        help="List of channels to be plotted, default is all",
    )
    parser.add_argument(
        "--selectionAxes",
        type=str,
        nargs="*",
        default=["charge", "passIso", "passMT", "cosThetaStarll", "qGen"],
        help="List of axes where for each bin a separate plot is created",
    )
    parser.add_argument(
        "--axlim",
        type=float,
        default=None,
        nargs="*",
        help="min and max for axes (2 values per axis)",
    )
    parser.add_argument(
        "--invertAxes",
        action="store_true",
        help="Invert the order of the axes when plotting",
    )
    parser.add_argument(
        "--chisq",
        type=str,
        default="automatic",
        choices=["automatic", "saturated", "linear", " ", "none", None],
        help="Type of chi2 to print on plot (saturated from fit likelihood. linear from observables, or none) 'automatic' means pick saturated for basemodel and otherwise linear",
    )
    parser.add_argument(
        "--dataName", type=str, default="Data", help="Data name for plot labeling"
    )
    parser.add_argument(
        "--xlabel", type=str, default=None, help="x-axis label for plot labeling"
    )
    parser.add_argument(
        "--ylabel", type=str, default=None, help="y-axis label for plot labeling"
    )
    parser.add_argument(
        "--processGrouping", type=str, default=None, help="key for grouping processes"
    )
    parser.add_argument(
        "--binSeparationLines",
        type=float,
        default=None,
        nargs="*",
        help="Plot vertical lines for makro bin edges in unrolled plots, specify bin boundaries to plot lines, if empty plot for all",
    )
    parser.add_argument(
        "--extraTextLoc",
        type=float,
        nargs="*",
        default=None,
        help="Location in (x,y) for additional text, aligned to upper left",
    )
    parser.add_argument(
        "--varNames", type=str, nargs="*", default=None, help="Name of variation hist"
    )
    parser.add_argument(
        "--varLabels",
        type=str,
        nargs="*",
        default=None,
        help="Label(s) of variation hist for plotting",
    )
    parser.add_argument(
        "--varColors",
        type=str,
        nargs="*",
        default=None,
        help="Color(s) of variation hist for plotting",
    )
    parser.add_argument(
        "--varOneSided",
        type=int,
        nargs="*",
        default=[],
        help="Only plot one sided variation (1) or two default two-sided (0)",
    )
    parser.add_argument(
        "--showVariations",
        type=str,
        default="lower",
        choices=["upper", "lower", "both"],
        help="Plot the variations in the upper, lower panels, or both",
    )
    parser.add_argument(
        "--scaleVariation",
        nargs="*",
        type=float,
        default=[],
        help="Scale a variation by this factor",
    )
    parser.add_argument(
        "--subplotSizes",
        nargs=2,
        type=int,
        default=[4, 2],
        help="Relative sizes for upper and lower panels",
    )
    parser.add_argument(
        "--correlatedVariations", action="store_true", help="Use correlated variations"
    )
    parser.add_argument(
        "--unfoldedXsec", action="store_true", help="Plot unfolded cross sections"
    )
    parser.add_argument(
        "--noPrefit",
        action="store_true",
        help="Don't plot prefit distribution",
    )
    parser.add_argument(
        "--noBinWidthNorm",
        action="store_true",
        help="Do not normalize bin yields by bin width",
    )
    parser.add_argument(
        "--upperPanelUncertaintyBand",
        action="store_true",
        help="Plot an uncertainty band in the upper panel around the prediction",
    )
    parser.add_argument(
        "--uncertaintyLabel",
        type=str,
        default=None,
        help="Label for uncertainty shown in the (ratio) plot",
    )
    parser.add_argument(
        "--fixed_param",
        type=str,
        default="time",
        help="the parameter that is not plotted. you choose a singe bin of this",
    )
    
    parser.add_argument(
        "--useData",
        type=str,
        default=False,
        help="whether to use data or mc for the heatmap",
    )
    
    parser.add_argument(
        "--annot", type = bool, default = False, help="display values in center of the cell"
    )
        
        
    
    
    args = parser.parse_args()

    return args


def make_plot(
    h_data,
    h_stack,
    axes,
    outdir,
    config,
    colors=None,
    labels=None,
    args=None,
    hup=None,
    hdown=None,
    h_data_stat=None,
    variation="",
    suffix="",
    chi2=None,
    meta=None,
    saturated_chi2=False,
    lumi=None,
    selection=None,
    fittype="postfit",
    varNames=None,
    varLabels=None,
    varColors=None,
    is_normalized=False,
    binwnorm=1.0,
    counts=True,
    cmap = "coolwarm"
):
        
    params = list(h_data.axes.name)
    varying_params = [p for p in params if p != args.fixed_param]
    varying_params_ind = [i for i in range(len(axes)) if axes[i].name in varying_params]

    axis_1 = [axes[varying_params_ind[0]].value(i) for i in range(len(axes[varying_params_ind[0]])+1)]
    axis_2 = [axes[varying_params_ind[1]].value(i) for i in range(len(axes[varying_params_ind[1]])+1)]
    axes_names = [a.name for a in axes] ## [time, pt_probe, eta_probe]

    ### NEED TO FIX THIS
    # pdb.set_trace()
    h_data = h_data[{f"{args.fixed_param}": 1}]
    if len(h_data.shape) > 2:
        flat = np.prod(h_data.shape[: len(h_data.shape) // 2])
        h_data = h_data.reshape((flat, flat))
        
    fig, ax = plt.subplots(figsize=(8, 6))
    # sns.color_palette("Spectral", as_cmap=True)

    
    mesh = ax.pcolormesh(axis_2, axis_1, h_data.values(), cmap='Spectral_r')

    # sns.heatmap(
    #     h_data.values(),
    #     cmap="Spectral_r",
    #     annot=args.annot,
    #     annot_kws={"fontsize":6},
    #     # fmt=".2g",
    #     cbar=True,
    #     linewidths=0.5,
    #     ax=ax,
    #     xticklabels = axis_2, 
    #     yticklabels = axis_1,
    #     fmt = ".3f"
    # )
    ax.set_aspect('auto')
    plt.colorbar(mesh, ax=ax, label='efficiency')


        
    outfile = f"{varying_params[0]}_{varying_params[1]}_{args.title}"



    plot_tools.add_decor(
        ax,
        outfile,
        args.subtitle,
        lumi=lumi,  # if args.dataName == "Data" and not args.noData else None,
        loc=args.titlePos,
        text_size=args.legSize,
    )
    
    ax.set_xlabel(varying_params[1])
    ax.set_ylabel(varying_params[0])

    if args.prefit:
        outfile += "_prefit"
    
    if args.useData == True:
        ax.set_title("Data")
        outfile += "_data"
    else:
        ax.set_title("MC") 
        outfile += "_mc"
    plot_tools.save_pdf_and_png(outdir, outfile)

    analysis_meta_info = None
    if meta is not None:
        if "meta_info_input" in meta:
            analysis_meta_info = {
                "RabbitOutput": meta["meta_info"],
                "AnalysisOutput": meta["meta_info_input"]["meta_info"],
            }
        else:
            analysis_meta_info = {"AnalysisOutput": meta["meta_info"]}

    output_tools.write_index_and_log(
        outdir,
        outfile,
        analysis_meta_info={
            **analysis_meta_info,
            },
            args=args,
        )


def make_plots(
    result,
    outdir,
    config,
    procs=None,
    labels=None,
    colors=None,
    args=None,
    channel="",
    lumi=1,
    fittype="postfit",
    varNames=None,
    varLabels=None,
    varColors=None,
    binwnorm=None,
    *opts,
    **kwopts,
):
    
    
    hist_data_stat = None
    if args.prefit:
        fittype = "prefit"
    if args.unfoldedXsec: ### should go through this
        if args.useData:
            hist = result[f"hist_{fittype}_inclusive"].get()
            name_impacts = f"hist_global_impacts_grouped_{fittype}_inclusive"
            if name_impacts in result.keys():
                hist_data_stat = result[name_impacts].get()[{"impacts": "stat"}]
        else: 
            hist = result[f"hist_prefit_inclusive"].get()
        hist_stack = []

        
    else:
        print("NEED TO DO CROSS SECTION")
        
    
    axes = [a for a in hist.axes]

    # vary poi by postfit uncertainty
    if varNames is not None:
        hist_var = result[
            f"hist_{fittype}_inclusive_variations{'_correlated' if args.correlatedVariations else ''}"
        ].get()
    else: ## goes in here, i should have a variation plot though
        hist_var = None

    if args.processGrouping is not None: ## this should be none
        hist_stack, labels, colors, procs = config.process_grouping(
            args.processGrouping, hist_stack, procs
        )

    labels = [
        l if p not in args.suppressProcsLabel else None for l, p in zip(labels, procs)
    ]

    if hist_var is not None:
        hists_down = [
            hist_var[{"downUpVar": 0, "vars": n}].project(*[a.name for a in axes])
            for n in varNames
        ]
        hists_up = [
            hist_var[{"downUpVar": 1, "vars": n}].project(*[a.name for a in axes])
            for n in varNames
        ]
    else: ### here
        hists_down = None
        hists_up = None

    if args.unfoldedXsec: ### i set this to 1
        # convert number in cross section in pb
        if lumi is not None and binwnorm is not None:
            to_xsc = lambda h: hh.scaleHist(h, 1.0 / (lumi * 1000))
        else:
            to_xsc = lambda h: h

        
        hist = to_xsc(hist)
        hist_stack = [to_xsc(h) for h in hist_stack]
        if hist_data_stat is not None:
            hist_data_stat = to_xsc(hist_data_stat)
        if hist_var is not None:
            hists_up = [to_xsc(h) for h in hists_up]
            hists_down = [to_xsc(h) for h in hists_down]

    # make plots in slices (e.g. for charge plus an minus separately)
    selection_axes = [a for a in axes if a.name in args.selectionAxes] ### i have none of these
    if len(selection_axes) > 0: ## this is 0
        selection_bins = [
            np.arange(a.size) for a in axes if a.name in args.selectionAxes
        ]
        other_axes = [a for a in axes if a not in selection_axes]

        ts = getattr(config, "translate_selection", {})

        for bins in itertools.product(*selection_bins):
            idxs = {a.name: i for a, i in zip(selection_axes, bins)}
            # next two dictionaries are built to print the bin values in the plot
            # if the axis name is not in the configuration dictionary, just use the bin index
            # (might print the two bin edges, but usually the bin index is enough)
            idxs_centers = {
                a.name: (
                    a.centers[i]
                    if isinstance(a, (hist.axis.Regular, hist.axis.Variable))
                    and a.name in ts
                    else a.edges[i] if a.name in ts else i
                )
                for a, i in zip(selection_axes, bins)
            }
            idxs_edges = {
                a.name: (
                    (a.edges[i], a.edges[i + 1])
                    if isinstance(a, (hist.axis.Regular, hist.axis.Variable))
                    else a.edges[i]
                )
                for a, i in zip(selection_axes, bins)
            }

            h = hist[idxs]
            h_stack = [h[idxs] for h in hist_stack]

            
            if hist_data_stat is not None:
                h_data_stat = hist_data_stat[idxs]
            else:
                h_data_stat = None

            if hists_up is not None:
                hup = [
                    (
                        h[{k.replace("Sig", ""): v for k, v in idxs.items()}]
                        if h is not None
                        else None
                    )
                    for h in hists_up
                ]
            else:
                hup = None

            if hists_down is not None:
                hdown = [h[idxs] if h is not None else None for h in hists_down]
            else:
                hdown = None

            selection = []
            for a in selection_axes:
                n = a.name
                if n in ts:
                    sel = ts.get(n, lambda x: f"{n}={x}")
                else:
                    sel = lambda x: f"{n} bin = {int(x)}"

                nparams = len(inspect.signature(sel).parameters)

                if nparams == 2:
                    selection.append(sel(*idxs_edges[n]))
                elif nparams == 1:
                    selection.append(sel(idxs_centers[n]))

            suffix = f"{channel}_" + "_".join(
                [
                    f"{a}_{str(i).replace('.','p').replace('-','m')}"
                    for a, i in idxs.items()
                ]
            )
            logger.info(
                f"Make plot for axes {[a.name for a in other_axes]}, in bins {idxs}"
            )
            make_plot(
                h,
                h_stack,
                other_axes,
                outdir,
                config,
                labels=labels,
                colors=colors,
                args=args,
                suffix=suffix,
                hup=hup,
                hdown=hdown,
                h_data_stat=h_data_stat,
                selection=selection,
                lumi=lumi,
                fittype=fittype,
                varNames=varNames,
                varLabels=varLabels,
                varColors=varColors,
                binwnorm=binwnorm,
                *opts,
                **kwopts,
            )
    else: #### okay go to this
        make_plot(
            hist,
            hist_stack,
            axes,
            outdir,
            config,
            labels=labels,
            colors=colors,
            args=args,
            suffix=channel,
            hup=hists_up,
            hdown=hists_down,
            h_data_stat=hist_data_stat,
            lumi=lumi,
            fittype=fittype,
            varNames=varNames,
            varLabels=varLabels,
            varColors=varColors,
            binwnorm=binwnorm,
            *opts,
            **kwopts,
        )


def get_chi2(result, no_chi2=True, fittype="postfit"):
    chi2_key = f"chi2_prefit" if fittype == "prefit" else "chi2"
    ndf_key = f"ndf_prefit" if fittype == "prefit" else "ndf"
    if not no_chi2 and fittype == "postfit" and result.get("postfit_profile", False):
        # use saturated likelihood test if relevant
        chi2 = 2.0 * result["nllvalreduced"]
        ndf = result["ndfsat"]
        return chi2, ndf, True
    elif not no_chi2 and chi2_key in result:
        return result[chi2_key], result[ndf_key], False
    else:
        return None, None, False


def main():
    args = parseArgs()
    global logger
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    config = plot_tools.load_config(args.config)

    varNames = args.varNames
    varLabels = args.varLabels
    varColors = args.varColors
    if varNames is not None:
        if varLabels is None:
            syst_labels = getattr(config, "systematics_labels", {})
            varLabels = [syst_labels.get(x, x) for x in varNames]
        elif len(varLabels) != len(varNames):
            raise ValueError(
                "Must specify the same number of args for --varNames, and --varLabels"
                f" found varNames={len(varNames)} and varLabels={len(varLabels)}"
            )
        if varColors is None:
            varColors = [
                colormaps["tab10" if len(varNames) < 10 else "tab20"](i)
                for i in range(len(varNames))
            ]

    fittype = "prefit" if args.prefit else "postfit"

    # load .hdf5 file first, must exist in combinetf and rabbit
    fitresult, meta = rabbit.io_tools.get_fitresult(args.infile, args.result, meta=True)

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

    outdir = output_tools.make_plot_dir(args.outpath, eoscp=args.eoscp)

    opts = dict(
        args=args,
        procs=procs,
        labels=labels,
        colors=colors,
        meta=meta,
        fittype=fittype,
        varNames=varNames,
        varLabels=varLabels,
        varColors=varColors,
    )

    results = fitresult["physics_models"]
    for margs in args.physicsModel:
        if margs == []:
            instance_keys = results.keys()
        else:
            model_key = " ".join(margs)
            instance_keys = [k for k in results.keys() if k.startswith(model_key)]
            if len(instance_keys) == 0:
                raise ValueError(f"No model found under {model_key}")

        for instance_key in instance_keys:

            is_normalized = any(
                instance_key.startswith(x) for x in ["Normalize", "Normratio"]
            )

            instance = results[instance_key]

            chi2, ndf, saturated_chi2 = get_chi2(
                (
                    fitresult
                    if fittype == "postfit"
                    and (
                        (instance_key == "Basemodel" and args.chisq != "linear")
                        or args.chisq == "saturated"
                    )
                    else instance
                ),
                args.chisq in [" ", "none", None],
                fittype,
            )

            for channel, result in instance["channels"].items():
                if args.channels is not None and channel not in args.channels:
                    continue
                logger.info(f"Make plot for {instance_key} in channel {channel}")

                info = channel_info.get(channel, {})

                suffix = f"{channel}_{instance_key}"
                for sign, rpl in [
                    (" ", "_"),
                    (".", "p"),
                    ("-", "m"),
                    (":", ""),
                    (",", ""),
                    ("slice(None)", ""),
                    ("(", ""),
                    (")", ""),
                    (":", ""),
                ]:
                    suffix = suffix.replace(sign, rpl)

                counts = not args.unfoldedXsec  # if histograms represent counts or not
                binwnorm = (
                    1.0
                    if any(
                        instance_key.startswith(x)
                        for x in ["Basemodel", "Project", "Select", "Norm"]
                    )
                    and not args.noBinWidthNorm
                    else None
                )

                opts["counts"] = counts

                make_plots(
                    result,
                    outdir,
                    config,
                    channel=suffix,
                    chi2=[chi2, ndf],
                    saturated_chi2=saturated_chi2,
                    lumi=info.get("lumi", None),
                    is_normalized=is_normalized,
                    binwnorm=binwnorm,
                    **opts,
                )

    if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
        output_tools.copy_to_eos(outdir, args.outpath)


if __name__ == "__main__":
    main()
