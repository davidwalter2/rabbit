"""Tests for --scanSaveDetail: what a likelihood scan records about its points.

Without it, ``nll_scan`` stores only the dnll curve. It restores the starting
vector when it finishes, so the parameter vectors it visited are gone, and the
only per-point convergence evidence is whatever scipy happened to print into
the log -- which is ``success``/``status`` (trust-krylov reports False at
perfectly converged points, so they carry no information) plus four of the
gradient entries. Recovering one point's parameters afterwards costs a whole
extra constrained fit.

These tests pin down that the opt-in path records the two things that fixes:
the full parameter vector per point, and a real edmval over the parameters the
scan point actually minimised.
"""

import tempfile

import numpy as np

from rabbit import fitter, inputdata
from rabbit.param_models.helpers import load_model

from .test_sparse_fit import make_options, make_test_tensor

SCAN_POINTS = 5
SCAN_RANGE = 1.0


def _fit(outdir):
    filename = make_test_tensor(outdir)
    indata_obj = inputdata.FitInputData(filename)
    param_model = load_model("Mu", indata_obj)
    f = fitter.Fitter(indata_obj, param_model, make_options())
    f.set_nobs(indata_obj.data_obs)
    f.minimize()
    # nll_scan scales its range by sqrt(cov[idx, idx]), so the covariance must exist
    val, grad, hess = f.loss_val_grad_hess()
    from rabbit.tfhelpers import edmval_cov

    _, cov = edmval_cov(grad, hess)
    f.cov.assign(cov)
    return f


def test_detail_off_records_nothing_extra():
    """The default path computes no detail and returns None in its place."""
    with tempfile.TemporaryDirectory() as outdir:
        f = _fit(outdir)
        scan_vals, dnlls, detail = f.nll_scan("sig", SCAN_RANGE, SCAN_POINTS)
        assert detail is None
        assert scan_vals.shape == dnlls.shape == (SCAN_POINTS,)


def test_detail_on_records_every_point_and_reproduces_the_curve():
    """The stored vectors must be the points the scan actually evaluated.

    This is the test that matters: re-evaluating the loss at each stored
    parameter vector has to give back the stored dnll. If it does, the vectors
    are the scan's own points and can be diffed against each other after the
    fact -- which is the whole purpose.
    """
    with tempfile.TemporaryDirectory() as outdir:
        f = _fit(outdir)
        nll_best = f.reduced_nll().numpy()

        scan_vals, dnlls, detail = f.nll_scan(
            "sig", SCAN_RANGE, SCAN_POINTS, save_detail=True
        )

        nparms = int(f.x.shape[0])
        assert detail["x"].shape == (SCAN_POINTS, nparms)
        assert detail["diagnostics"].shape == (
            SCAN_POINTS,
            len(f.SCAN_DETAIL_FIELDS),
        )
        assert np.all(np.isfinite(detail["x"]))

        # nll_scan restored the starting vector, so this also checks it did
        xval_after = f.x.numpy()

        idx = int(np.where(f.parms.astype(str) == "sig")[0][0])
        for i in range(SCAN_POINTS):
            # the scanned parameter sits at the scan value in the stored vector
            assert np.isclose(detail["x"][i, idx], scan_vals[i], atol=1e-12)
            f.x.assign(detail["x"][i])
            assert np.isclose(f.reduced_nll().numpy() - nll_best, dnlls[i], atol=1e-8)
        f.x.assign(xval_after)


def test_edmval_is_small_at_converged_scan_points():
    """A converged profile point has EDM ~0 over the parameters it minimised."""
    with tempfile.TemporaryDirectory() as outdir:
        f = _fit(outdir)
        _, _, detail = f.nll_scan("sig", SCAN_RANGE, SCAN_POINTS, save_detail=True)

        fields = list(f.SCAN_DETAIL_FIELDS)
        edm = detail["diagnostics"][:, fields.index("edmval")]
        gmax = detail["diagnostics"][:, fields.index("grad_max_abs")]

        # the central point runs no fit of its own; the off-central ones do
        off_central = [i for i in range(SCAN_POINTS) if i != SCAN_POINTS // 2]
        assert np.all(np.abs(edm[off_central]) < 1e-6), edm
        assert np.all(gmax[off_central] < 1e-4), gmax


def test_edmval_only_sees_the_floating_parameters():
    """The full-space CG edmval at a scan point is the floating-subspace EDM.

    The scanned parameter is frozen, and frozen parameters enter the loss
    through ``tf.stop_gradient``: their gradient entry and Hessian row/column
    are exactly zero, so the CG solve never leaves the floating subspace. Check
    that premise away from the minimum, where the EDM is not trivially ~0, by
    comparing against the dense EDM of the floating block.
    """
    with tempfile.TemporaryDirectory() as outdir:
        f = _fit(outdir)
        idx = int(np.where(f.parms.astype(str) == "sig")[0][0])
        f.freeze_params("sig")
        x = f.x.numpy().copy()
        x[idx] += 0.5 * np.sqrt(f.cov[idx, idx].numpy())
        f.x.assign(x)
        f.minimize()
        # push the floating parameters off the profile minimum
        x = f.x.numpy().copy()
        x[f.floating_indices] += 0.1
        f.x.assign(x)

        _, grad, hess = f.loss_val_grad_hess()
        assert grad.numpy()[idx] == 0.0
        assert np.all(hess.numpy()[idx] == 0.0)

        edm_dense, _ = f.edmval_cov(grad, hess)
        edm_cg = f.edmval_cov_rows_hessfree(grad, [])[0]
        assert edm_dense > 1e-6
        assert np.isclose(edm_cg, edm_dense, rtol=1e-8)
        f.defreeze_params("sig")
