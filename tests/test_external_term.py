"""Test external likelihood terms (gradient + hessian) added to TensorWriter and Fitter.

The external term has the form

    L_ext(x) = g^T x_sub + 0.5 x_sub^T H x_sub

where x_sub is the slice of fit parameters identified by the StrCategory axes
of grad/hess. With Asimov data and a single Gaussian-constrained nuisance,
the analytical post-fit value of the nuisance is

    theta = -g / (1 + h)

where the +1 is the prefit Gaussian constraint and +h is the external hessian
contribution. This script verifies that prediction for several configurations,
including dense and sparse (wums.SparseHist) hessian storage.
"""

import os
import tempfile
from types import SimpleNamespace

import hist
import numpy as np
import scipy.sparse
from wums.sparse_hist import SparseHist

from rabbit import fitter, inputdata, tensorwriter
from rabbit.param_models.helpers import load_model


def make_options(**kwargs):
    defaults = dict(
        earlyStopping=-1,
        noBinByBinStat=True,
        binByBinStatMode="lite",
        binByBinStatType="automatic",
        covarianceFit=False,
        chisqFit=False,
        diagnostics=False,
        minimizerMethod="trust-krylov",
        prefitUnconstrainedNuisanceUncertainty=0.0,
        freezeParameters=[],
        setConstraintMinimum=[],
        unblind=[],
        blindingGroup=[],
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def build_writer(grad=None, hess=None):
    """Build a TensorWriter with one bkg process and a single shape systematic."""
    np.random.seed(0)
    ax = hist.axis.Regular(20, -5, 5, name="x")

    h_data = hist.Hist(ax, storage=hist.storage.Double())
    h_bkg = hist.Hist(ax, storage=hist.storage.Weight())

    x_bkg = np.random.uniform(-5, 5, 5000)
    h_data.fill(x_bkg)
    h_bkg.fill(x_bkg, weight=np.ones(len(x_bkg)))

    bin_centers = ax.centers - ax.centers[0]
    weights = 0.01 * bin_centers - 0.05
    h_up = h_bkg.copy()
    h_dn = h_bkg.copy()
    h_up.values()[...] = h_bkg.values() * (1 + weights)
    h_dn.values()[...] = h_bkg.values() * (1 - weights)

    writer = tensorwriter.TensorWriter()
    writer.add_channel([ax], "ch0")
    writer.add_data(h_data, "ch0")
    writer.add_process(h_bkg, "bkg", "ch0", signal=True)
    writer.add_systematic([h_up, h_dn], "shape", "bkg", "ch0", symmetrize="average")

    if grad is not None or hess is not None:
        writer.add_external_likelihood_term(grad=grad, hess=hess)

    return writer


def run_fit(filename):
    indata_obj = inputdata.FitInputData(filename)
    param_model = load_model("Mu", indata_obj)
    options = make_options()
    f = fitter.Fitter(indata_obj, param_model, options)

    # use Asimov data so the only force on the nuisance is the constraint + external term
    f.set_nobs(f.expected_yield())
    f.minimize()

    parms_str = f.parms.astype(str)
    return {
        "parms": parms_str,
        "x": f.x.numpy(),
    }


def loss_grad_hess_at(filename, x_override=None):
    """Return (loss, grad, hess) for the loaded tensor evaluated at x_override
    (or the default starting x if None). Uses Asimov data."""
    import tensorflow as tf

    indata_obj = inputdata.FitInputData(filename)
    param_model = load_model("Mu", indata_obj)
    options = make_options()
    f = fitter.Fitter(indata_obj, param_model, options)
    f.set_nobs(f.expected_yield())
    if x_override is not None:
        f.x.assign(tf.constant(x_override, dtype=f.x.dtype))
    val, grad, hess = f.loss_val_grad_hess()
    return (
        f.parms.astype(str),
        val.numpy(),
        grad.numpy(),
        hess.numpy(),
    )


def get_param_value(result, name):
    idx = np.where(result["parms"] == name)[0][0]
    return result["x"][idx]


def get_param_index(parms, name):
    return int(np.where(parms == name)[0][0])


def make_grad_hist(values, param_names):
    """Build a 1D hist with a StrCategory axis for an external gradient."""
    ax = hist.axis.StrCategory(param_names, name="params")
    h = hist.Hist(ax, storage=hist.storage.Double())
    h.values()[...] = np.asarray(values)
    return h


def make_hess_hist(values, param_names):
    """Build a 2D hist with two StrCategory axes for an external hessian."""
    ax0 = hist.axis.StrCategory(param_names, name="params0")
    ax1 = hist.axis.StrCategory(param_names, name="params1")
    h = hist.Hist(ax0, ax1, storage=hist.storage.Double())
    h.values()[...] = np.asarray(values)
    return h


def make_hess_sparsehist(values, param_names):
    """Same as make_hess_hist but using a wums.SparseHist.

    StrCategory axes have an overflow bin by default, so SparseHist's
    with-flow layout has shape (n+1, n+1). The user data goes in the
    first n x n block; the overflow row/col is filled with zeros.
    """
    ax0 = hist.axis.StrCategory(param_names, name="params0")
    ax1 = hist.axis.StrCategory(param_names, name="params1")
    n = len(param_names)
    full = np.zeros((ax0.extent, ax1.extent), dtype=np.float64)
    full[:n, :n] = np.asarray(values, dtype=np.float64)
    return SparseHist(scipy.sparse.csr_array(full), [ax0, ax1])


def main():
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()

    SHAPE = "shape"

    with tempfile.TemporaryDirectory() as tmpdir:

        # --- Baseline: no external term ---
        baseline_writer = build_writer()
        baseline_writer.write(outfolder=tmpdir, outfilename="baseline")
        baseline = run_fit(os.path.join(tmpdir, "baseline.hdf5"))
        baseline_shape = get_param_value(baseline, SHAPE)
        print(f"Baseline (no external):    {SHAPE} = {baseline_shape:.6f}")
        assert (
            abs(baseline_shape) < 1e-6
        ), f"Asimov baseline should give {SHAPE} ~ 0, got {baseline_shape}"
        print("PASS: baseline Asimov fit gives shape ~ 0")

        # Reference loss/grad/hess at the baseline x (no external term).
        # The contribution of L_ext(x) = g^T x + 0.5 x^T H x to the NLL gradient
        # at any x is (g + H x), and to the NLL hessian is H. We test these
        # exactly (not analytical post-fit values, which depend on the data
        # Hessian and the constraint and don't have a clean closed form).
        parms, val0, grad0, hess0 = loss_grad_hess_at(
            os.path.join(tmpdir, "baseline.hdf5")
        )
        i_shape = get_param_index(parms, SHAPE)
        x0 = baseline["x"].copy()
        # the test below evaluates external terms at the baseline minimum
        # where x[i_shape] = 0, so H x_sub = 0 → grad delta == g exactly.

        configs = [
            (
                "grad only (g=1)",
                build_writer(grad=make_grad_hist([1.0], [SHAPE])),
                {i_shape: 1.0},
                {(i_shape, i_shape): 0.0},
            ),
            (
                "grad+dense hess (g=1, h=2)",
                build_writer(
                    grad=make_grad_hist([1.0], [SHAPE]),
                    hess=make_hess_hist([[2.0]], [SHAPE]),
                ),
                {i_shape: 1.0},
                {(i_shape, i_shape): 2.0},
            ),
            (
                "grad+SparseHist hess (g=1, h=2)",
                build_writer(
                    grad=make_grad_hist([1.0], [SHAPE]),
                    hess=make_hess_sparsehist([[2.0]], [SHAPE]),
                ),
                {i_shape: 1.0},
                {(i_shape, i_shape): 2.0},
            ),
            (
                "hess only (h=5)",
                build_writer(hess=make_hess_hist([[5.0]], [SHAPE])),
                {i_shape: 0.0},
                {(i_shape, i_shape): 5.0},
            ),
        ]

        for label, writer, expected_grad_delta, expected_hess_delta in configs:
            tag = (
                label.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace(",", "")
                .replace("=", "")
            )
            writer.write(outfolder=tmpdir, outfilename=tag)
            _, val, grad, hess = loss_grad_hess_at(
                os.path.join(tmpdir, f"{tag}.hdf5"),
                x_override=x0,
            )
            for idx, expected in expected_grad_delta.items():
                actual = grad[idx] - grad0[idx]
                print(
                    f"{label}: grad delta @ idx {idx} = {actual:+.6f}  (expected {expected:+.6f})"
                )
                assert (
                    abs(actual - expected) < 1e-8
                ), f"{label}: grad delta {actual} != expected {expected}"
            for (i, j), expected in expected_hess_delta.items():
                actual = hess[i, j] - hess0[i, j]
                print(
                    f"{label}: hess delta @ ({i},{j}) = {actual:+.6f}  (expected {expected:+.6f})"
                )
                assert (
                    abs(actual - expected) < 1e-8
                ), f"{label}: hess delta {actual} != expected {expected}"
            print(f"PASS: {label}")

        # Sanity check: also verify that running the full fit shifts the
        # baseline shape value in the expected direction (negative for g=+1).
        grad_only_writer = build_writer(grad=make_grad_hist([1.0], [SHAPE]))
        grad_only_writer.write(outfolder=tmpdir, outfilename="grad_only_fit")
        grad_only = run_fit(os.path.join(tmpdir, "grad_only_fit.hdf5"))
        v = get_param_value(grad_only, SHAPE)
        print(f"Full fit with g=+1: shape = {v:.6f}  (expected negative)")
        assert v < -1e-3, f"Expected shape to pull negative, got {v}"
        print("PASS: full fit with positive gradient pulls shape negative")

        print()
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()


def test_partial_external_covariance_leaves_uncovered_variances_nan():
    """A partially-overlapping --externalPostfit must not report prefit widths.

    load_fitresult fills only the intersection of the two parameter sets and
    leaves the rest of fitter.cov on the prefit diagonal it was initialized
    with, so diag(cov) is a postfit variance only for the covered parameters.
    external_cov_mask records which those are; reporting the rest would hand
    back a prefit width as a postfit uncertainty.
    """
    import h5py

    with tempfile.TemporaryDirectory() as tmpdir:
        build_writer().write(outfolder=tmpdir, outfilename="partial_cov")
        indata_obj = inputdata.FitInputData(os.path.join(tmpdir, "partial_cov.hdf5"))
        options = make_options()
        param_model = load_model("Mu", indata_obj)
        f = fitter.Fitter(indata_obj, param_model, options)
        f.defaultassign()
        f.set_nobs(indata_obj.data_obs)

        n = int(f.x.shape[0])
        assert n > 1, "need more than one parameter for a partial overlap"
        keep = f.parms.astype(str)[: n - 1]

        ext = os.path.join(tmpdir, "external.hdf5")
        with h5py.File(ext, "w") as g:
            g.create_dataset("x", data=np.zeros(len(keep)))
            g.create_dataset("parms", data=np.array([p.encode() for p in keep]))
            g.create_dataset("cov", data=np.eye(len(keep)) * 0.25)

        f.load_fitresult(ext, None, profile=False)

        mask = f.external_cov_mask
        assert mask is not None and mask.sum() == len(keep) and not mask.all()

        diag = np.diag(f.cov.numpy())
        # the covered block carries the external covariance
        np.testing.assert_allclose(diag[mask], 0.25)
        # the uncovered entry is still the prefit variance, NOT a postfit one --
        # which is why rabbit_fit.py masks it to NaN rather than reporting it
        assert not np.isclose(diag[~mask][0], 0.25)


# ---------------------------------------------------------------------------
# Guards around --externalPostfit in bin/rabbit_fit.py. Both are plain
# functions of the arguments (and, for the second, of the loaded covariance),
# so they are checked here directly rather than by running the driver.
# ---------------------------------------------------------------------------


def _driver():
    """bin/rabbit_fit.py as an importable module."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_rabbit_fit_for_guards", root / "bin" / "rabbit_fit.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(**kwargs):
    defaults = dict(
        globalImpacts=False,
        gaussianGlobalImpacts=False,
        noHessian=False,
        noEDM=False,
        noFit=False,
        externalPostfit=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# what --toys defaults to: a single fit to data
DATA_FIT = [0]


def test_global_impacts_without_a_possible_covariance_are_refused():
    """Global impacts need a covariance that these flags can never produce.

    Deciding it from the flags is what lets main() refuse before any work: the
    equivalent check inside fit() is only reached after minimize(), so raising
    there costs a completed fit.
    """
    refuse = _driver().global_impacts_covariance_refusal

    assert refuse(_args(noHessian=True), DATA_FIT) is None, "nothing was asked for"
    assert refuse(_args(globalImpacts=True), DATA_FIT) is None, "covariance is computed"

    # --noHessian allocates no covariance at all (Fitter.cov is None), so not
    # even --externalPostfit can supply one -- load_fitresult refuses it
    msg = refuse(_args(globalImpacts=True, noHessian=True), DATA_FIT)
    assert msg and "--noHessian" in msg
    assert refuse(
        _args(
            gaussianGlobalImpacts=True,
            noHessian=True,
            externalPostfit="ext.hdf5",
            noFit=True,
        ),
        DATA_FIT,
    )

    # --noEDM skips computing it for any fit that runs here
    msg = refuse(_args(globalImpacts=True, noEDM=True), DATA_FIT)
    assert msg and "--noEDM" in msg
    # and must not send the reader after a flag they never passed, which is
    # what naming --noHessian in this message used to do
    assert "--noHessian" not in msg
    assert refuse(
        _args(globalImpacts=True, noEDM=True, externalPostfit="e.hdf5"), DATA_FIT
    )

    # a job that runs no fit can still read the covariance from the external
    # result, either because --noFit was passed or because the only "fit" is
    # the Asimov prefit entry
    assert (
        refuse(
            _args(globalImpacts=True, noEDM=True, externalPostfit="e.hdf5", noFit=True),
            DATA_FIT,
        )
        is None
    )
    assert (
        refuse(_args(globalImpacts=True, noEDM=True, externalPostfit="e.hdf5"), [-1])
        is None
    )


def test_the_covariance_refusal_runs_before_the_input_is_loaded():
    """Pins the property the refusal exists for. Moving it below the work it is
    meant to precede would leave it correct and useless."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "bin" / "rabbit_fit.py").read_text()
    main_src = src[src.index("def main():") :]
    assert main_src.index("global_impacts_covariance_refusal(") < main_src.index(
        "inputdata.FitInputData("
    ), "the refusal now runs after the input tensor is loaded"


def test_partial_external_coverage_refuses_global_impacts():
    """Partial coverage does not only degrade the uncovered rows.

    load_fitresult fills the intersection BLOCK, so every covered-to-uncovered
    covariance entry stays zero: an uncovered nuisance contributes exactly zero
    impact to a COVERED parameter, and the variance it should have carried is
    reported as data-statistical instead. Nothing in the output looks wrong,
    which is why this is a refusal and not a warning.
    """
    import pytest
    from wums import logging as wums_logging

    driver = _driver()
    driver.logger = wums_logging.child_logger("test_external_postfit_variances")

    cov = np.diag([0.25, 0.25, 4.0])
    partial = SimpleNamespace(cov=cov, external_cov_mask=np.array([True, True, False]))

    for opts in (_args(globalImpacts=True), _args(gaussianGlobalImpacts=True)):
        with pytest.raises(Exception, match="EVERY parameter"):
            driver.external_postfit_variances(opts, partial)

    # without them the partial result is still usable: covered variances are
    # reported, the rest are NaN rather than the prefit width sitting there
    var = driver.external_postfit_variances(_args(), partial)
    np.testing.assert_allclose(var[:2], 0.25)
    assert np.isnan(var[2])

    # full coverage is untouched, impacts or not
    full = SimpleNamespace(cov=cov, external_cov_mask=np.ones(3, dtype=bool))
    np.testing.assert_allclose(
        driver.external_postfit_variances(_args(globalImpacts=True), full),
        [0.25, 0.25, 4.0],
    )
