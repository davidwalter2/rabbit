#!/usr/bin/env python3

import argparse
import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from rabbit import io_tools
from wums import logging, output_tools, plot_tools
from rabbit import debugdata, inputdata

logger = logging.child_logger(__name__)

sort_choices = []
sort_choices_abs = [f"abs {s}" for s in sort_choices]


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        default=os.path.expanduser("./test"),
        help="Base path for output",
    )

    parser.add_argument(
        "--inputDir",
        type=str,
        help="Directory containing fitresults_xx.hdf5 files (xx = exactly two characters, e.g. cu, dd)",
    )
    parser.add_argument(
        "--keepNuisances",
        type=str,
        default=None,
        help="Match nuisances by regular expression",
    )
    parser.add_argument(
        "--excludeNuisances",
        type=str,
        default=None,
        help="Exclude nuisances by regular expression",
    )
        
    parser.add_argument(
        "--result",
        default=None,
        type=str,
        help="fitresults key in file (e.g. 'asimov'). Leave empty for data fit result.",
    )
   
    parser.add_argument(
        "--nPoi",
        type=int,
        default=-1,
        help="number of pois to plot",
    )
    parser.add_argument(
        "--quark",
        type=str,
        default=None,
        choices=["u", "d", "s"],
        help="Only plot the 8 coefficients (c and d) for this quark flavour",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        default=None,
        help="Only use these fitresults files, given as the part between 'fitresults_' and '.hdf5' or as file name, e.g. '--files cu dd', '--files tu_rapidity_c' or '--files fitresults_cu.hdf5'",
    )
    parser.add_argument(
        "--rapidity",
        action="store_true",
        help="Use fitresults_xx_rapidity.hdf5 files instead of fitresults_xx.hdf5",
    )
    parser.add_argument(
        "--noLiterature",
        action="store_true",
        help="Do not plot the literature (ZEUS) values",
    )
    parser.add_argument(
        "--xScale",
        type=float,
        default=1e-6,
        help="Factor the fitted values and uncertainties are multiplied by before plotting",
    )
    parser.add_argument(
        "--xlim",
        type=float,
        default=None,
        help="Symmetric x-axis range [-xlim, xlim] (after scaling). Default: automatic from the 95%% CL intervals",
    )
    parser.add_argument(
        "--poiFormat",
        type=bool,
        default=True,
        help="formatting for my specific poi",
    )
    parser.add_argument(
        "-p", "--postfix", type=str, default="", help="Postfix for output file name"
    )
    return parser.parse_args()



# file suffix is <coefficient type><quark flavour>, e.g. "cu" -> cxxu, cxyu, cxzu, cyzu
FILE_ORDER = ["cu", "cd", "cs", "du", "dd", "ds"]


def get_fitresult_files(input_dir, quark=None, tags=None, rapidity=False):
    suffix = "_rapidity" if rapidity else ""
    FILE_PATTERN = re.compile(rf"^fitresults_(..){suffix}\.hdf5$")
    if tags is not None:
        # explicitly requested files: accept a file name ("fitresults_tu_rapidity_c.hdf5")
        # or the part between "fitresults_" and ".hdf5" ("cu", "tu_rapidity_c"),
        # with the rapidity suffix appended if that file exists
        filenames, missing = [], []
        for t in tags:
            t = os.path.basename(t)
            candidates = [t] if t.endswith(".hdf5") else [f"fitresults_{t}{suffix}.hdf5", f"fitresults_{t}.hdf5"]
            found = [os.path.join(input_dir, c) for c in candidates if os.path.isfile(os.path.join(input_dir, c))]
            if not found:
                missing.append(t)
                continue
            if quark is not None:
                match = re.match(r"^fitresults_.(.)", os.path.basename(found[0]))
                if match and match.group(1) != quark:
                    continue
            filenames.append(found[0])
        if missing:
            raise FileNotFoundError(f"No fitresults files for {missing} found in {input_dir}")
        if not filenames:
            raise FileNotFoundError(f"None of the requested files {tags} match quark '{quark}'")
        return filenames
    files = {}
    for f in glob.glob(os.path.join(input_dir, f"fitresults_*{suffix}.hdf5")):
        match = FILE_PATTERN.match(os.path.basename(f))
        if not match:
            continue
        tag = match.group(1)
        if quark is not None and tag[1] != quark:
            continue
        files[tag] = f
    if not files:
        quark_str = "" if quark is None else f" for quark '{quark}'"
        raise FileNotFoundError(f"No fitresults_xx{suffix}.hdf5 files{quark_str} found in {input_dir}")
    order = {k: i for i, k in enumerate(FILE_ORDER)}
    return [files[k] for k in sorted(files, key=lambda k: (order.get(k, len(order)), k))]


def get_fit_values(filenames, result=None, keep_nuisances=None, exclude_nuisances=None):
    """Return {poi: [value, uncertainty]} collected from all fitresult files"""
    fit_values = {}
    for filename in filenames:
        logger.info(f"Reading {filename}")
        fitresult, meta = io_tools.get_fitresult(filename, result, meta=True)
        pois = set(io_tools.get_poi_names(meta))
        labels, pulls, constraints = io_tools.get_pulls_and_constraints(
            fitresult,
            keep_nuisances=keep_nuisances,
            exclude_nuisances=exclude_nuisances,
        )
        for label, pull, constraint in zip(labels, pulls, constraints):
            label = str(label)
            if label not in pois:
                continue
            if label in fit_values:
                logger.warning(f"{label} found in multiple files, using {filename}")
            fit_values[label] = [pull, constraint]
    return fit_values


def main():
    args = parseArgs()

    filenames = get_fitresult_files(args.inputDir, args.quark, args.files, args.rapidity)
    fit_values = get_fit_values(
        filenames,
        args.result,
        keep_nuisances=args.keepNuisances,
        exclude_nuisances=args.excludeNuisances,
    )

    labels = list(fit_values.keys())
    pulls = np.array([v[0] for v in fit_values.values()])
    constraints = np.array([v[1] for v in fit_values.values()])

    n = len(labels) if args.nPoi < 0 else min(args.nPoi, len(labels))
    labels, pulls, constraints = labels[:n], pulls[:n], constraints[:n]

    literature_vals = {"cxyu": [-32e-5, 4.1e-5], "cxzu": [-5.4e-4, 1.4e-4], "cyzu": [-3.7e-4, 2.1e-4], "cxxu": [-2.1e-5, 2.4e-5], "cxyd": [-16e-4, 2.0e-4], "cxzd": [-27e-4, 7e-4], "cyzd": [-1.8e-3, 1e-3], "cxxd": [-1.0e-3, 1.2e-3], "cxys": [-26e-4, 3.3e-4], "cxzs": [-4.4e-3, 1.2e-3], "cyzs": [-3e-3, 1.7e-3], "cxxs": [-1.7e-3, 2.e-3], "ctxu": [-25e-5, 6.6e-5], "ctyu": [-17e-5, 9.8e-5], "ctxd": [-7.8e-4, 2e-4], "ctyd": [-5.2e-4, 3.0e-4], "ctxs": [-9.6e-4, 2.5e-4], "ctys": [-6.4e-4, 3.7e-4]
    }
    # y positions (top to bottom)
    y = np.arange(n)[::-1]
    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * n + 1)))
    # 95% CL (thin lines)
    ax.errorbar(pulls*args.xScale, y, xerr=constraints*2*args.xScale, fmt='none',
                ecolor='black', elinewidth=1, capsize=0,)
    # 68% CL (thicker lines + points)
    ax.errorbar(pulls*args.xScale, y, xerr=constraints*args.xScale, fmt='o',
                color='black', ecolor='black', elinewidth=3, capsize=0, label='68%, 95% CL')
    # literature values: [lower, upper] half-widths of the 68% CL interval around 0,
    # plotted directly on the (already scaled) x axis
    lit_extent = 0
    if not args.noLiterature:
        previously_labeled = False
        for i, label in enumerate(labels):
            if label not in literature_vals or literature_vals[label][0] == 0:
                continue
            this_val = np.abs(np.array(literature_vals[label]))[:, None]
            lit_extent = max(lit_extent, 2 * this_val.max())
            ax.errorbar(0, y[i] + 0.25, xerr=this_val, color='C0', elinewidth=3, capsize=0,
                        label=None if previously_labeled else 'ZEUS \n 68%, 95% CL')
            ax.errorbar(0, y[i] + 0.25, xerr=2 * this_val, color='C0', fmt='none', elinewidth=1, capsize=0)
            previously_labeled = True
    # zero reference line
    ax.axvline(0, color='black', linestyle='--', linewidth=1)

    # labels & ticks
    ax.set_yticks(y)
    if args.poiFormat:
        labels_mod = [f'${l[0]}^{{{l[1:3]}}}_{l[-1]}$' for l in labels]
        ax.set_yticklabels(labels_mod)
    else:
        ax.set_yticklabels(labels)

    
    # ax.set_xlabel("SME coefficient value")
    # group separators (optional)
    # separate the groups of 4 coefficients coming from each file
    for sep in np.arange(4, n, 4):
        ax.axhline(n - sep - 0.5, color='gray', linewidth=1)
    xlim = args.xlim
    if xlim is None:
        xlim = 1.1 * max(np.max(np.abs(pulls) + 2 * constraints) * abs(args.xScale), lit_extent)
    ax.set_xlim([-xlim, xlim])
    # grid (vertical only)
    ax.xaxis.grid(True, linestyle=':', color='gray', alpha=0.7)
    ax.yaxis.grid(False)
    ax.yaxis.minorticks_off()
    ax.legend(frameon=True, fontsize = 16)

    # cleaner look
    ax.spines['top'].set_visible(True)
    # ax.spines['right'].set_visible(True)

    
    outdir = output_tools.make_plot_dir(args.outpath)

    # indata = inputdata.FitInputData(args.infile, pseudodata=args.pseudodata)
    
    outfile = 'fitted_values' + args.postfix
    plot_tools.save_pdf_and_png(outdir, outfile)
    output_tools.write_index_and_log(
            outdir,
            outfile,
            args=args,
        )
    
    logname = f"{outdir}/{outfile}.log"
    
    
    with open(logname, "a") as logf:
        logf.write("\n")
        logf.write("-------------------------------------------- \n")
        logf.write("process: # of events\n")
    
         

if __name__ == "__main__":
    main()
