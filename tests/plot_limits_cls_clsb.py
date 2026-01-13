import argparse

import numpy as np
from scipy.stats import norm

from wums import output_tools, plot_tools, logging  # isort: skip

logger = logging.setup_logger(__file__, 4, False)

parser = argparse.ArgumentParser()
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
parser.add_argument("--useTF", action="store_true", help="Use tf (default is numpy)")
parser.add_argument("--case", type=int, default=0, help="Case to be plotted")
args = parser.parse_args()

outdir = output_tools.make_plot_dir(args.outpath)

case = args.case

if case == 0:
    xobs = -0.09795380612182193
    xobs_err = 0.04407941867588671
    xerr = 0.044638215763924

    limit = 0.05518237377499425

elif case == 1:
    xobs = 0.889939508709423
    xobs_err = 0.04939466327844699
    xerr = 0.04468240951622258

    limit = 0.9711864997550228
elif case == 2:
    xobs = 0.0005802774284884333
    xobs_err = 0.0446372204041607
    xerr = 0.04463908622809369

    limit = 0.08787167868041947

cl_s = 0.05


###
if xobs < 0:
    # modified statistics
    def qmu(x):
        return ((x - xobs) / xobs_err) ** 2 - (xobs / xobs_err) ** 2

else:

    def qmu(x):
        return ((x - xobs) / xobs_err) ** 2


def qA(x):
    return (x / xerr) ** 2


if args.useTF:
    import tensorflow as tf

    from rabbit import tfhelpers as tfh

    logger.info("Use tf")

    ### using tf
    xobs = tf.constant(xobs, dtype=tf.float64)
    xobs_err = tf.constant(xobs_err, dtype=tf.float64)
    xerr = tf.constant(xerr, dtype=tf.float64)
    cl_s = tf.constant(cl_s, dtype=tf.float64)

    def phi_qmu(x):
        qmu_sqrt = tf.sqrt(qmu(x))
        return tfh.normal_cdf(-qmu_sqrt)

    def phi_qmu_qA(x):
        qmu_sqrt = tf.sqrt(qmu(x))
        qA_sqrt = tf.sqrt(qA(x))
        return tfh.normal_cdf(qA_sqrt - qmu_sqrt)

    def f(x):
        return phi_qmu(x) - cl_s * phi_qmu_qA(x)

    def fprime(x):

        xvar = tf.constant(x, dtype=tf.float64)
        with tf.GradientTape() as t:
            t.watch(xvar)
            val = f(xvar)
        grad = t.gradient(val, xvar)

        return grad

else:
    logger.info("Use np")

    def phi_qmu(x):
        qmu_sqrt = np.sqrt(qmu(x))
        return norm.cdf(-qmu_sqrt)

    def phi_qmu_qA(x):
        qmu_sqrt = np.sqrt(qmu(x))
        qA_sqrt = np.sqrt(qA(x))
        return norm.cdf(qA_sqrt - qmu_sqrt)

    def f(x):
        return phi_qmu(x) - cl_s * phi_qmu_qA(x)

    def fprime(x):
        qmu_sqrt = np.sqrt(qmu(x))
        qA_sqrt = np.sqrt(qA(x))

        # PDFs
        phi_qmu = norm.pdf(qmu_sqrt)
        phi_qA_qmu = norm.pdf(qA_sqrt - qmu_sqrt)

        # Derivatives of qmu and qA
        dqmu_dx = (x - xobs) / (qmu_sqrt * xobs_err**2)
        dqA_dx = 1.0 / xerr

        return -phi_qmu * dqmu_dx - cl_s * phi_qA_qmu * (dqA_dx - dqmu_dx)


name = "tf" if args.useTF else "np"
name += f"_case{case}"

xlim = [0.0, limit * 1.5]
x = np.linspace(*xlim, 100)

fig, ax1 = plot_tools.figure(
    None,
    r"$\mu$",
    r"$-2\Delta \ln(L)$",
    width_scale=1,
    xlim=xlim,
    ylim=[0, 50],
    automatic_scale=False,
)

y1 = qmu(x)
y2 = qA(x)
ax1.plot(x, y1, color="red", marker="", linestyle="--", label=r"$q_\mu$")
ax1.plot(x, y2, color="blue", marker="", linestyle="--", label="$q_A$")

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

# plot_tools.fix_axes(ax1, ax2, fig)#, yscale=args.yscale, logy=args.logy)

to_join = [name, "qmu_qA"]
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

ylim = [-0.1, 0.5]
rrange = [-0.5, 0.1]
fig, ax1, ratio_axes = plot_tools.figureWithRatio(
    None,
    r"$\mu$",
    r"$-2\Delta \ln(L)$",
    width_scale=1,
    xlim=xlim,
    ylim=ylim,
    automatic_scale=False,
    rlabel="",
    rrange=rrange,
    subplotsizes=[4, 2],
)
ax2 = ratio_axes[0]

ax1.plot(
    x,
    f(x),
    color="black",
    marker="",
    label=r"$\Phi(\sqrt{q_\mu}) - CL_s \Phi(\sqrt{q_A} - \sqrt{q_\mu})$",
)

ax1.plot(
    x, phi_qmu(x), color="red", marker="", linestyle="--", label=r"$\Phi(\sqrt{q_\mu})$"
)
ax1.plot(
    x,
    cl_s * phi_qmu_qA(x),
    color="blue",
    marker="",
    linestyle=":",
    label=r"$CL_s \Phi(\sqrt{q_A} - \sqrt{q_\mu})$",
)


ax1.plot(xlim, [0, 0], color="grey", marker="", linestyle="--", label=None)

ax1.plot([limit, limit], ylim, color="grey", marker="", linestyle="--", label=None)

# ax2.plot(x, norm.cdf(-np.sqrt(y1)), color="red", marker="", label=r"$\Phi(\sqrt{q_\mu})$")
# ax2.plot(x, cl_s * norm.cdf(np.sqrt(y2)-np.sqrt(y1)), color="blue", marker="", label=r"$CL_s \Phi(\sqrt{q_\mu})$")

ax2.plot(x, fprime(x), color="black", marker="", label=r"$df(\mu)/dx$")

ax2.plot([limit, limit], rrange, color="grey", marker="", linestyle="--", label=None)
ax2.plot(xlim, [0, 0], color="grey", marker="", linestyle="--", label=None)

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

plot_tools.addLegend(
    ax2,
    ncols=args.legCols,
    loc="lower right",
)

plot_tools.fix_axes(ax1, ax2, fig)  # , yscale=args.yscale, logy=args.logy)

to_join = [name, "cls"]
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
