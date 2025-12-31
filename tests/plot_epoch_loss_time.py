import argparse

import numpy as np

import rabbit.io_tools

from wums import output_tools, plot_tools, logging  # isort: skip

logger = logging.setup_logger(__file__, 4, False)

parser = argparse.ArgumentParser()
parser.add_argument(
    "infile",
    type=str,
    nargs="+",
    help="hdf5 file from rabbit or root file",
)
parser.add_argument(
    "--labels",
    type=str,
    nargs="+",
    help="Label for each input file",
)
parser.add_argument(
    "--result",
    default=None,
    type=str,
    help="fitresults key in file (e.g. 'asimov'). Leave empty for data fit result.",
)
parser.add_argument("--logy", action="store_true", help="Make the yscale logarithmic")
parser.add_argument("-p", "--postfix", type=str, help="Postfix for output file name")
parser.add_argument("-o", "--outpath", default="./", help="output directory")
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
    "--legCols", type=int, default=2, help="Number of columns in legend"
)
parser.add_argument("--startEpoch", type=int, default=0, help="Epoch to start plotting")
args = parser.parse_args()

outdir = output_tools.make_plot_dir(args.outpath)

epochs = []
times = []
losses = []
dlosses = []
for infile in args.infile:
    fitresult, meta = rabbit.io_tools.get_fitresult(infile, args.result, meta=True)

    h_time = fitresult["epoch_time"].get()
    h_loss = fitresult["epoch_loss"].get()

    times.append(h_time.values())
    loss = 2 * h_loss.values()

    epochs.append(np.arange(1, len(loss) + 1))

    losses.append(loss)
    dlosses.append(-np.diff(loss))  # reduction of loss after each epoch

linestyles = [
    "-",
    "--",
    ":",
    "-.",
    "-",
    "--",
    ":",
    "-.",
]
linestyles = linestyles[: len(epochs)]

start = args.startEpoch
stop = None

for x, y, xlabel, ylabel, stop, suffix in (
    (times, losses, "time [s]", r"$-2\Delta \ln(L)$", None, "loss"),
    (epochs, losses, "epoch", r"$-2\Delta \ln(L)$", None, "loss_time"),
    (times, dlosses, "epoch", r"$-2(\ln(L_{t}) - \ln(L_{t-1}))$", -1, "reduction_loss"),
    (
        epochs,
        dlosses,
        "time [s]",
        r"$-2(\ln(L_{t}) - \ln(L_{t-1}))$",
        -1,
        "reduction_loss_time",
    ),
):
    ymin = min([min(iy) for iy in y])

    y = [iy - ymin for iy in y]
    x = [ix[start:stop] for ix in x]

    if args.logy:
        ylim = [
            min([min(iy[iy > 0]) for iy in y]) * 0.5,
            max([max(iy) for iy in y]) * 2,
        ]
    else:
        ylim = [0, max([max(iy) for iy in y]) * 1.1]

    fig, ax1 = plot_tools.figure(
        None,
        xlabel,
        ylabel,
        width_scale=1,
        xlim=[0, max([max(ix) for ix in x])],
        ylim=ylim,
        automatic_scale=False,
        logy=args.logy,
    )

    for ix, iy, l, s in zip(x, y, args.labels, linestyles):
        ax1.plot(ix, iy, label=l, linestyle=s)

    plot_tools.add_decor(
        ax1,
        args.title,
        args.subtitle,
        data=True,
        lumi=None,
        loc=args.titlePos,
        no_energy=True,
    )

    plot_tools.addLegend(
        ax1,
        ncols=args.legCols,
        loc=args.legPos,
    )

    # plot_tools.fix_axes(ax1, fig, logy=args.logy)

    name = f"epoch_{suffix}"
    to_join = [name, args.postfix]
    outfile = "_".join(filter(lambda x: x, to_join))
    if args.subtitle == "Preliminary":
        outfile += "_preliminary"

    plot_tools.save_pdf_and_png(outdir, outfile)
    analysis_meta_info = None
    output_tools.write_index_and_log(
        outdir,
        outfile,
        args=args,
    )
