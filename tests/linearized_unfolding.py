import argparse
import copy

import hist
import numpy as np
from scipy.stats import norm, truncnorm

from rabbit import tensorwriter

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", default="./", help="output directory")
parser.add_argument("--outname", default="test_tensor", help="output file name")
parser.add_argument(
    "--postfix",
    default=None,
    type=str,
    help="Postfix to append on output file name",
)
parser.add_argument(
    "--systematicType",
    choices=["log_normal", "normal"],
    default="log_normal",
    help="probability density for systematic variations",
)
parser.add_argument(
    "--combineProcesses",
    default=False,
    action="store_true",
    help="If processes should be combined into one",
)
args = parser.parse_args()


n_bins = 10
n_proc = 5
n_events = 200  # number of events per process
n_systematics = 10
n_pseudodata = 1000

# true total yield per process
expected_yields = np.ones(n_proc) * n_events

# true signal modifiers
true_mu = np.array([0.9, 0.95, 1, 1.05, 1.1])

observed_yields = true_mu * expected_yields

bin_edges = np.linspace(0, 1, n_bins + 1)
bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2.0


def get_pred(mu, sigma):
    probs = norm.cdf(bin_edges[1:], loc=mu, scale=sigma) - norm.cdf(
        bin_edges[:-1], loc=mu, scale=sigma
    )
    pred = probs * n_events / np.sum(probs)
    return pred


preds = [get_pred(i / (n_proc - 1), 0.2) for i in range(n_proc)]


# generate random systematic uncertainties
def sinus_basis(frequency, phase):
    return lambda x, f=frequency, p=phase: np.sin(2 * np.pi * f * (x + p))


def random_normal(sigma, mu=0, lo=-0.1, hi=0.1, size=1):
    a_, b_ = (lo - mu) / sigma, (hi - mu) / sigma
    rv = truncnorm(a_, b_, loc=mu, scale=sigma)
    return rv.rvs(size=size)


# define systematic uncertainties
systematics = {}
for i in range(n_systematics):
    frequency = np.random.uniform(0.25, n_bins / 2.0)
    phase = np.random.uniform(0, 1)
    basis = sinus_basis(frequency, phase)

    for pred in preds:
        # fully correlated across all processes,
        #   the detla is random meaning that each process is affected differently
        delta = random_normal(sigma=0.02, mu=0, lo=-0.1, hi=0.1)[0]

        systematics[f"syst_{i}"] = lambda x, d=delta, f=basis: d * f(x)


ax_x = hist.axis.Regular(10, 0, 1, name="x")
ax_masked = hist.axis.Regular(5, 0, 1, name="x")

# Build tensor
writer = tensorwriter.TensorWriter(
    systematic_type=args.systematicType,
)

writer.add_channel([ax_x], "ch0")

observed = np.sum(preds * true_mu[:, None], axis=0)

writer.add_data(observed, "ch0")  # add true as data

# add pseudodata
for i in range(n_pseudodata):
    theta = {name: np.random.normal(0, 1) for name in systematics}

    central = 0
    for j, pred in enumerate(preds):

        pred_obs = pred * true_mu[j]

        # norm uncertainties
        vars = [
            (1 + syst(bin_centers)) ** theta[name] for name, syst in systematics.items()
        ]
        pred_obs = pred_obs * np.prod(vars, axis=0)

        central += pred_obs

    counts = np.random.poisson(central)

    writer.add_pseudodata(counts, f"p{i}", "ch0")

# to make comparisons in parameter independent ways
writer.add_channel([ax_masked], "ch0_masked", masked=True)

# we make 2 models (hence 2 writers), one where we split processes and one where we merge them
writer_per_process = copy.deepcopy(writer)

# one inclusive process as background (constrained normalization)
pred_inclusive = np.sum(preds, axis=0)
pred_masked = np.sum(preds, axis=1)
writer.add_process(pred_inclusive, f"proc", "ch0", signal=False)
writer.add_process(pred_masked, f"proc", "ch0_masked")

# we now add the variations of individual processes as systematic variations
for i, pred in enumerate(preds):
    pred_var = pred_inclusive + pred * 0.1

    writer.add_systematic(
        pred_var,
        f"proc{i}",
        f"proc",
        "ch0",
        constrained=False,
    )

    pred_masked_var = pred_masked.copy()
    pred_masked_var[i] = pred_masked[i] + np.sum(pred * 0.1)
    writer.add_systematic(
        pred_masked_var,
        f"proc{i}",
        f"proc",
        "ch0_masked",
        constrained=False,
    )

for i, pred in enumerate(preds):
    writer_per_process.add_process(pred, f"proc{i}", "ch0", signal=True)
    pred_masked = np.zeros(n_proc)
    pred_masked[i] = np.sum(pred)
    writer_per_process.add_process(pred_masked, f"proc{i}", "ch0_masked", signal=True)

# and now the real systematic variations
for syst_name, syst in systematics.items():
    var = 1 + syst(bin_centers)

    preds_var = preds * var

    writer.add_systematic(
        np.sum(preds_var, axis=0),
        syst_name,
        f"proc",
        "ch0",
    )

    for i, pred_var in enumerate(preds_var):
        writer_per_process.add_systematic(
            pred_var,
            syst_name,
            f"proc{i}",
            "ch0",
        )

directory = args.output
if directory == "":
    directory = "./"
filename = f"{args.outname}_{args.systematicType}"
if args.postfix:
    filename += f"_{args.postfix}"
writer.write(outfolder=directory, outfilename=filename)
writer_per_process.write(outfolder=directory, outfilename=f"{filename}_per_process")
