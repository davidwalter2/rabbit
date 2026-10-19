"""Shared toy builders and fit driver of the limited-MC-statistics
de-biasing tests (tests/test_mcstat_*.py).

The tests are split over several files by topic: each file runs in its own
pytest process in CI, because XLA keeps the code it compiles for every fitter
mapped for the life of the process and one process for all of them would
exceed vm.max_map_count.

The toy builders write into pytest's tmp_path and are parameterised by the
number of folds ``k``, ``sparse`` and ``systematic_type`` so further
configurations can reuse them.
"""

import functools
import importlib.util
import os
from pathlib import Path

import hist
import numpy as np

from rabbit import fitter, inputdata
from rabbit.param_models import helpers as ph
from rabbit.tensorwriter import TensorWriter

REPO = Path(__file__).resolve().parents[1]

# The near-degenerate 2-process toy of the MC-stat studies: a flat process and a
# "step" process equal to it in the 2nd half and +3% in the 1st half, with
# finite (1:1) MC templates.
DEGEN_NBINS = 200
DEGEN_FLAT, DEGEN_STEP_HI, DEGEN_STEP_LO = 4962.0, 5112.0, 4962.0
DEGEN_SEED = 20240614

# POI directions: the degenerate difference and the well-constrained sum.
DDIF = np.array([1.0, -1.0]) / np.sqrt(2.0)

# Linear POIs, Gaussian (chisq) likelihood, no bin-by-bin stat: the setting in
# which the closed-form numpy references apply.
LINEAR_CHISQ = ("--chisqFit", "--noBinByBinStat", "--allowNegativeParam")


# ---------------------------------------------------------------------------
# fit driver
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _driver():
    """bin/rabbit_fit.py as an importable module (for its make_parser())."""
    spec = importlib.util.spec_from_file_location(
        "_rabbit_fit_for_mcstat_tests", REPO / "bin" / "rabbit_fit.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_toy(filename, outdir, *args):
    """Build and minimize a Fitter as `rabbit_fit.py filename -t 0 *args` would.

    Returns the minimized Fitter; callers take the loss/grad/hess and the
    de-biased covariances from it.
    """
    options = (
        _driver()
        .make_parser()
        .parse_args([str(filename), "-o", str(outdir), "-t", "0", *args])
    )
    indata = inputdata.FitInputData(str(filename), None)
    param_model = ph.load_models([["Mu"]], indata, **vars(options))
    f = fitter.Fitter(indata, param_model, options, do_blinding=False)
    f.defaultassign()
    f.set_nobs(indata.data_obs)
    f.minimize()
    return f


def sigma(cov, direction=DDIF):
    return float(np.sqrt(direction @ np.asarray(cov) @ direction))


# ---------------------------------------------------------------------------
# toy builders
# ---------------------------------------------------------------------------


def _hist1d(ax, values, variances=None):
    h = hist.Hist(ax, storage=hist.storage.Weight())
    h.view()["value"] = values
    h.view()["variance"] = values if variances is None else variances
    return h


def _fold_hist(ax, folds):
    """[k, nbins] fold counts -> hist with an `mcfold` axis (unit weights)."""
    k = folds.shape[0]
    axf = hist.axis.IntCategory(list(range(k)), name="mcfold")
    h = hist.Hist(ax, axf, storage=hist.storage.Weight())
    for i in range(k):
        h.view()["value"][:, i] = folds[i]
        h.view()["variance"][:, i] = folds[i]
    return h


def split_into_folds(rng, counts, k):
    """Assign each bin's integer counts to k folds, multinomial(1/k): [k, nbins]."""
    return rng.multinomial(counts, [1.0 / k] * k).T


def _write(tw, path, name):
    tw.write(outfolder=str(path), outfilename=name)
    return os.path.join(str(path), name)


def degenerate_toy_arrays(k=None, seed=DEGEN_SEED):
    """True yields, Asimov data and one finite-MC realization of the degenerate
    toy; with `k`, also each process's split into k folds.

    Returns a dict with ``n_true`` [2, nb], ``data`` [nb], ``sumw`` [2, nb]
    (unit-weight counts, so sumw2 == sumw) and ``folds`` [2, k, nb] (or None).
    """
    nb = DEGEN_NBINS
    rng = np.random.default_rng(seed)
    n_flat = np.full(nb, DEGEN_FLAT)
    n_step = np.concatenate(
        [np.full(nb // 2, DEGEN_STEP_HI), np.full(nb // 2, DEGEN_STEP_LO)]
    )
    sumw = np.stack([rng.poisson(n_flat), rng.poisson(n_step)])
    folds = None
    if k is not None:
        folds = np.stack([split_into_folds(rng, c, k) for c in sumw])
    return dict(
        n_true=np.stack([n_flat, n_step]),
        data=n_flat + n_step,
        sumw=sumw.astype(float),
        folds=folds,
    )


def make_mcstat_moment_toy(path, with_M=True, bbb_variance=False, seed=DEGEN_SEED):
    """Degenerate toy with plain finite-MC templates and, if `with_M`, the frozen
    noise-floor matrix M over the two POIs (continuous-M de-biasing).

    M_jj = sum_b sumw2[b, j] / var_b with var_b = data_b (chisq), or with the
    BB-lite-inflated var_b = data_b + sum_proc sumw2[b] if `bbb_variance`.
    Returns (filename, arrays) with ``arrays["M"]`` set (None without M).
    """
    arr = degenerate_toy_arrays(seed=seed)
    ax = hist.axis.Regular(DEGEN_NBINS, 0.0, 1.0, name="x")
    data, sumw = arr["data"], arr["sumw"]
    tw = TensorWriter(sparse=False)
    tw.add_channel([ax], "ch0")
    tw.add_data(_hist1d(ax, data), "ch0")
    tw.add_process(_hist1d(ax, sumw[0]), "flat", "ch0", signal=True)
    tw.add_process(_hist1d(ax, sumw[1]), "step", "ch0", signal=True)
    arr["M"] = None
    if with_M:
        var = data + sumw.sum(0) if bbb_variance else data
        arr["M"] = np.diag(np.sum(sumw / var, axis=1))
        tw.add_mc_stat_moment(
            arr["M"],
            ["flat", "step"],
            variance="bb_inflated" if bbb_variance else "bare",
        )
    name = f"toy_{'M' if with_M else 'noM'}{'_bb' if bbb_variance else ''}.hdf5"
    return _write(tw, path, name), arr


# ---------------------------------------------------------------------------
# numpy references
# ---------------------------------------------------------------------------


def _fisher(T, V):
    """sum_b T[b,i] T[b,j] / V[b] for templates T [nbins, nparams]."""
    return np.einsum("bi,b,bj->ij", T, 1.0 / V, T)


# ---------------------------------------------------------------------------
# small well-conditioned fold toy (any k, additive systematics, sparse storage)
# ---------------------------------------------------------------------------

SMALL_NBINS = 60


def make_small_fold_toy(
    path, k, sparse=False, systematic_type="log_normal", syst=0, exempt=False
):
    """Well-conditioned 2-process toy with k MC-stat folds and optionally one
    SHARED (unfolded) shape systematic on `flat`, and (exempt) a third,
    unfolded background with zero MC variance (MC-stat-exempt, e.g.
    data-driven). Returns (filename, arrays)."""
    rng = np.random.default_rng(3)
    nb = SMALL_NBINS
    n_flat = np.full(nb, 400.0)
    n_step = np.linspace(100.0, 700.0, nb)
    folds = np.stack(
        [
            split_into_folds(rng, rng.poisson(n), k).astype(float)
            for n in (n_flat, n_step)
        ]
    )  # [2, k, nb]
    data = n_flat + n_step
    ax = hist.axis.Regular(nb, 0.0, 1.0, name="x")
    tw = TensorWriter(sparse=sparse, systematic_type=systematic_type)
    tw.add_channel([ax], "ch0")
    tw.add_data(_hist1d(ax, data), "ch0")
    for name, f in zip(("flat", "step"), folds):
        tw.add_process(_fold_hist(ax, f), name, "ch0", signal=True, fold_axis="mcfold")
    svar = None
    if syst:
        # additive variation (absolute, = logk for systematic_type "normal")
        svar = 0.05 * folds[0].sum(0) * np.linspace(-1.0, 1.0, nb)
        tw.add_systematic(
            _hist1d(ax, folds[0].sum(0) + svar), "s0", "flat", "ch0", mirror=True
        )
    if exempt:
        tw.add_process(
            _hist1d(ax, np.full(nb, 150.0), variances=np.zeros(nb)), "exempt", "ch0"
        )
    fname = f"small_k{k}_{int(sparse)}_{systematic_type}_{syst}_{int(exempt)}.hdf5"
    return _write(tw, path, fname), dict(folds=folds, data=data, svar=svar)
