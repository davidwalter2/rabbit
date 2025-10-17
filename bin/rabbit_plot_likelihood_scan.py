#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from rabbit import io_tools

from wums import logging, output_tools, plot_tools  # isort: skip

hep.style.use(hep.style.ROOT)

logger = None


def writeOutput(fig, outfile, extensions=[], postfix=None, args=None, meta_info=None):
    name, _ = os.path.splitext(outfile)

    if postfix:
        name += f"_{postfix}"

    for ext in extensions:
        if ext[0] != ".":
            ext = "." + ext
        output = name + ext
        print(f"Write output file {output}")
        plt.savefig(output)

        output = name.rsplit("/", 1)
        output[1] = os.path.splitext(output[1])[0]
        if len(output) == 1:
            output = (None, *output)
    if args is None and meta_info is None:
        return
    output_tools.write_logfile(
        *output,
        args=args,
        meta_info=meta_info,
    )


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputFile",
        type=str,
        help="fitresults output",
    )
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
        "--result",
        default=None,
        type=str,
        help="fitresults key in file (e.g. 'asimov'). Leave empty for data fit result.",
    )
    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        default="./test",
        help="Folder path for output",
    )
    parser.add_argument(
        "-p", "--postfix", type=str, help="Postfix for output file name"
    )
    parser.add_argument(
        "--params",
        type=str,
        nargs="*",
        default=[],
        help="Parameters to plot the likelihood scan",
    )
    parser.add_argument(
        "--noHessian",
        action="store_true",
        help="Don't include Hessian likelihood approximation in plot",
    )
    parser.add_argument(
        "--combine",
        type=str,
        default=None,
        help="Provide a root file with likelihood scan result from combine",
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
        "--config",
        type=str,
        default=None,
        help="Path to config file for style formatting",
    )
    return parser.parse_args()


def plot_scan(
    h_scan,
    h_contours=None,
    param_value=0,
    param_variance=1,
    param="x",
    title=None,
    subtitle=None,
    titlePos=0,
    combine=None,
    ylabel=r"$-2\,\Delta \log L$",
    config={},
    no_hessian=False,
):

    xlabel = config.get("systematics_labels", {}).get(param, param)

    x = np.array(h_scan.axes["scan"]).astype(float)
    y = h_scan.values() * 2

    fig, ax = plot_tools.figure(
        x,
        xlabel,
        ylabel,
        xlim=(min(x), max(x)),
        ylim=(min(y), max(y)),  # logy=args.logy
    )

    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=4, color="gray", linestyle="--", alpha=0.5)

    if not no_hessian:
        parabola_vals = param_value + np.linspace(
            -3 * param_variance**0.5, 3 * param_variance**0.5, 100
        )
        parabola_nlls = 1 / param_variance * (parabola_vals - param_value) ** 2
        ax.plot(
            parabola_vals,
            parabola_nlls,
            marker="",
            markerfacecolor="none",
            color="red",
            linestyle="-",
            label="Hessian",
        )

    ax.plot(
        x,
        y,
        marker=None,  # "x",
        color="black",
        label="Likelihood scan" if combine is None else "Rabbit",
        markeredgewidth=2,
        linewidth=2,
    )

    if combine is not None:
        ax.plot(
            *combine,
            marker="o",
            linestyle=None,
            linewidth=0,
            color="orange",
            label="Combine",
        )

    if h_contours is not None:
        for i, cl in enumerate(h_contours.axes["confidence_level"]):
            x = h_contours[{"confidence_level": cl}].values()[::-1] + param_value
            y = np.full(len(x), float(cl) ** 2)
            label = "Contour scan" if i == 0 else None
            ax.plot(
                x,
                y,
                marker="o",
                markerfacecolor="none",
                color="black",
                linestyle="",
                markeredgewidth=2,
                label=label,
            )
            logger.info(f"{int(float(cl))} sigma confidence level for {param} in {x}")

            ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
            for ix in x:
                ax.axvline(x=ix, color="gray", linestyle="--", alpha=0.5)

    ax.legend(loc="upper right")

    plot_tools.add_decor(
        ax, title, subtitle, data=True, lumi=None, loc=titlePos, no_energy=True
    )

    return fig


def main():
    args = parseArgs()

    global logger
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    fitresult, meta = io_tools.get_fitresult(args.inputFile, args.result, meta=True)

    config = plot_tools.load_config(args.config)

    meta = {
        "rabbit": meta["meta_info"],
    }

    h_params = fitresult["parms"].get()

    if "contour_scan" in fitresult.keys():
        h_contour = fitresult["contour_scan"].get()
    else:
        h_contour = None

    parms = h_params.axes["parms"] if len(args.params) == 0 else args.params

    if args.combine is not None:
        import uproot

        with uproot.open(args.combine) as tfile:
            vals = tfile["limit"]["r"].array()
            nlls = tfile["limit"]["deltaNLL"].array()

            order = np.argsort(vals)
            vals = vals[order]
            nlls = nlls[order] * 2  # -2ln(L)

    for param in parms:
        p = h_params[{"parms": param}]
        param_value = p.value
        param_variance = p.variance
        h_scan = fitresult[f"nll_scan_{param}"].get()

        if h_contour is not None:
            h_contour_param = h_contour[{"parms": param, "impacts": param}]
        else:
            h_contour_param = None

        fig = plot_scan(
            h_scan,
            h_contour_param,
            param_value=param_value,
            param_variance=param_variance,
            param=param,
            title=args.title,
            subtitle=args.subtitle,
            titlePos=args.titlePos,
            combine=(vals, nlls) if args.combine is not None else None,
            config=config,
            no_hessian=args.noHessian,
        )
        os.makedirs(args.outpath, exist_ok=True)
        outfile = os.path.join(args.outpath, f"nll_scan_{param}")
        writeOutput(
            fig,
            outfile=outfile,
            extensions=["png", "pdf"],
            meta_info=meta,
            args=args,
            postfix=args.postfix,
        )


if __name__ == "__main__":
    main()
