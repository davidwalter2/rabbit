import argparse
import glob
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib import colormaps
from matplotlib.ticker import MaxNLocator
from wums import logging

import rabbit.io_tools

from wums import output_tools, plot_tools  # isort: skip
from wums import boostHistHelpers as hh  # isort: skip



hep.style.use(hep.style.ROOT)

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
parser.add_argument("-p", "--postfix", type=str, help="Postfix for output file name")
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
    "--ylim",
    type=float,
    nargs=2,
    help="Min and max values for y axis (if not specified, range set automatically)",
)
# parser.add_argument(
#     "--begin",
#     type=int,
#     default=0,
#     help="Begin",
# )
# parser.add_argument(
#     "--end",
#     type=int,
#     default=99,
#     help="End",
# )
parser.add_argument("--xlim", type=float, nargs=2, help="min and max for x axis")
parser.add_argument(
    "--rrange",
    type=float,
    nargs=2,
    default=[0.9, 1.1],
    help="y range for ratio plot",
)
parser.add_argument(
    "--invertAxes",
    action="store_true",
    help="Invert the order of the axes when plotting",
)
parser.add_argument("--ptMin", default=None, type=int, help="Lowest pT Bin to include")
parser.add_argument("--ptMax", default=None, type=int, help="Highest pT Bin to include")
parser.add_argument("--channel", default="ch0_masked", type=str, help="Channel")
args = parser.parse_args()
logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

# path = "/ceph/submit/data/user/d/david_w/Unfolding/rabbit_results/260123_unfolding/w_z/Combination_ZMassDileptonWMass_iteration*_forIteration/fitresults*.hdf5"
# path = "/scratch/submit/cms/dwalter/rabbit_results/260128_unfolding/w_z/Combination_ZMassDileptonWMass_iteration*_forIteration/fitresults*.hdf5"
path = "/scratch/submit/cms/dwalter/rabbit_results/260203_unfolding/w_z/Combination_ZMassDileptonWMass_iteration*_forIteration/fitresults_regularize_shapes.hdf5"

infiles = glob.glob(path)

channel = args.channel

# fitresult = rabbit.io_tools.get_fitresult(path.replace("*", "0").replace("fitresults0","fitresults"))
# hist_initial = fitresult["mappings"]["Select"]["channels"][channel]["hist_prefit_inclusive"].get()

iteration_numbers = []
hists_noi = []
for infile in infiles:
    logger.info(f"Now at {infile}")
    fitresult = rabbit.io_tools.get_fitresult(infile)
    if any(
        x in infile
        for x in [
            "for_w",
            "for_z",
            "regularize_strength50.hdf5",
            "Combination_ZMassDileptonWMass_iteration13_unconstrained_forIteration",
            "9_forIteration/fitresults.hdf5",
            "10_forIteration/fitresults.hdf5",
            "11_forIteration/fitresults.hdf5",
            "12_forIteration/fitresults.hdf5",
            "13_forIteration/fitresults.hdf5",
            "14_forIteration/fitresults.hdf5",
        ]
    ):
        logger.info(f"Don't want, skip")
        continue
    if "mappings" not in fitresult.keys():
        logger.info(f"No mapping, skip")
        continue

    iteration = int(infile.split("iteration")[1].split("/")[0].split("_")[0])
    logger.info(f"Load iteration {iteration} from {infile}")
    iteration_numbers.append(iteration)

    hist_postfit = fitresult["mappings"]["Select"]["channels"][channel][
        "hist_postfit_inclusive"
    ].get()

    hist_prefit = fitresult["mappings"]["Select"]["channels"][channel][
        "hist_prefit_inclusive"
    ].get()
    hist_noi = hh.divideHists(hist_postfit, hist_prefit)
    # hist_noi = hh.divideHists(hist_postfit, hist_initial)
    hists_noi.append(hist_noi)

x = np.array(iteration_numbers)


idx = np.argsort(x)
x = x[idx]

logger.info(f"Found iterations {x}")

hists_noi = [hists_noi[i] for i in idx]

xlim = (x.min() - 0.5, x.max() + 0.5)
# loop over changes
for iq in [0, 1]:
    # loop over eta bins

    if channel == "ch0_masked":
        if iq == 1:
            # no change for Z
            continue
        outdir = output_tools.make_plot_dir(f"{args.outpath}/Z/")
        nbins = 10
    else:
        outdir = output_tools.make_plot_dir(f"{args.outpath}/W_qGen{iq}/")
        nbins = 17

    for ieta in range(0, nbins):
        if channel == "ch0_masked":
            selection = {"absYVGen": ieta}
            name = f"noi_vs_iteration_absYVGen{ieta}"
        else:
            selection = {"qGen": iq, "absEtaGen": ieta}
            name = f"noi_vs_iteration_absEtaGen{ieta}"

        yy = np.array([h_noi[selection].values(flow=True) for h_noi in hists_noi])
        yy = np.transpose(yy)

        fig, ax = plot_tools.figure(yy, xlabel="iteration", ylabel="NOI", xlim=xlim)

        ax.plot(xlim, [1, 1], linestyle="--", color="grey")
        # ax.plot([9,9], [yy.min(), yy.max()], linestyle="--", color="grey") # first iteration with only constraining W NOIs
        # ax.plot([11,11], [yy.min(), yy.max()], linestyle="--", color="grey") # doubling size of variation
        # ax.plot([14,14], [yy.min(), yy.max()], linestyle="--", color="grey") # use regularization

        for i, y in enumerate(yy[args.ptMin : args.ptMax]):
            ibin = i + args.ptMin if args.ptMin is not None else i
            ax.plot(x, y, marker="o", label=f"pT bin {ibin}")

        if "ch1" in channel:
            ax.text(0.7, 0.2, f"q={'-1' if iq==0 else '+1'}", transform=ax.transAxes)

        var = r"\eta" if "ch1" in channel else "Y"
        ax.text(0.7, 0.1, rf"$|{var}|$ bin {ieta}", transform=ax.transAxes)

        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(ncols=2, loc="upper right")

        to_join = [name, args.postfix]
        outfile = "_".join(filter(lambda x: x, to_join))
        if args.subtitle == "Preliminary":
            outfile += "_preliminary"

        plot_tools.save_pdf_and_png(outdir, outfile)

        output_tools.write_index_and_log(
            outdir,
            outfile,
            args=args,
        )

        # exit()

exit()
for i in range(0, 13):
    for c in [0, 1]:

        h_prefit = []
        h_postfit = []
        # for i in range(args.begin, args.end):
        ifile = infile.replace("iteration0", f"iteration{i}")
        fitresult = rabbit.io_tools.get_fitresult(ifile)

        hi = fitresult["mappings"]["Select"]["channels"][channel][
            "hist_prefit_inclusive"
        ].get()[{"qGen": c, "absEtaGen": 17}]
        hf = fitresult["mappings"]["Select"]["channels"][channel][
            "hist_postfit_inclusive"
        ].get()[{"qGen": c, "absEtaGen": 17}]

        axes_names = hi.axes.name
        if len(axes_names) > 1:
            if args.invertAxes:
                axes_names = axes_names[::-1]
                axes = axes[::-1]
            hi = hh.unrolledHist(hi, binwnorm=None, obs=axes_names)
            hf = hh.unrolledHist(hf, binwnorm=None, obs=axes_names)

        h_prefit.append(hi)
        h_postfit.append(hf)

        outdir = output_tools.make_plot_dir(args.outpath)

        fig, ax1, ratio_axes = plot_tools.figureWithRatio(
            h_prefit[0],
            "bin",
            r"$\sigma$",
            args.ylim,
            "ratio",
            args.rrange,
            xlim=args.xlim,
            width_scale=1.25 if len(axes_names) == 1 else 1,
            # automatic_scale=args.customFigureWidth is None,
            # subplotsizes=args.subplotSizes,
            # logy=args.logy,
        )
        ax2 = ratio_axes[-1]

        varColors = [
            colormaps["tab10" if len(h_prefit) < 10 else "tab20"](j)
            for j in range(len(h_prefit))
        ]

        for j, (hi, hf) in enumerate(zip(h_prefit, h_postfit)):
            hep.histplot(
                hi,
                xerr=False,
                yerr=False,
                histtype="step",
                # density=args.density,
                binwnorm=None,
                ax=ax1,
                zorder=1,
                flow="none",
                label=i + j,
                color=varColors[j],
            )
            hep.histplot(
                hf,
                xerr=False,
                yerr=False,
                histtype="step",
                # density=args.density,
                binwnorm=None,
                ax=ax1,
                zorder=1,
                flow="none",
                linestyle="dashed",
                color=varColors[j],
            )

            hep.histplot(
                hh.divideHists(hf, hi),
                xerr=False,
                yerr=False,
                histtype="step",
                # density=args.density,
                binwnorm=None,
                ax=ax2,
                zorder=1,
                flow="none",
                linestyle="dashed",
                color=varColors[j],
            )

            hep.histplot(
                hh.divideHists(hf, hf),
                xerr=False,
                yerr=False,
                histtype="step",
                # density=args.density,
                binwnorm=None,
                ax=ax2,
                zorder=1,
                flow="none",
                linestyle="dashed",
                color="grey",
            )
        plot_tools.addLegend(
            ax1,
            ncols=args.legCols,
            loc=args.legPos,
            text_size=args.legSize,
            # extra_text=text_pieces if not args.noExtraText else None,
            # extra_text_loc=None if args.extraTextLoc is None else args.extraTextLoc[:2],
            # padding_loc=args.legPadding,
        )

        plot_tools.fix_axes(ax1, ax2, fig)

        to_join = [f"iterative_unfolding_{i}", f"charge{c}", args.postfix]
        outfile = "_".join(filter(lambda x: x, to_join))
        if args.subtitle == "Preliminary":
            outfile += "_preliminary"

        plot_tools.save_pdf_and_png(outdir, outfile)

        output_tools.write_index_and_log(
            outdir,
            outfile,
            args=args,
        )
