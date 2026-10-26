"""Frozen parameters under the dense-Hessian minimizers.

A frozen parameter is held by tf.stop_gradient alone: its gradient entry and
its Hessian row and column are exactly zero, but the loss value still depends
on it. trust-exact and tf-trust-exact used to move it anyway -- the singular
Hessian sends the More-Sorensen iteration into its hard case, which steps
along the frozen coordinate, and the step is accepted because the loss goes
down. A likelihood scan then came out flat (2dNLL ~ 0 where the profile is up
to 10) with every point at maxiter; dogleg's Cholesky failed on the same
Hessian and its scan points were not minimised at all. These tests pin down
that the scanned parameter stays exactly at the scan value and that the
profile is trust-krylov's, which never moved it.
"""

import tempfile

import numpy as np
import pytest

from rabbit import fitter, inputdata
from rabbit.param_models.helpers import load_model
from rabbit.tfhelpers import edmval_cov

from .test_sparse_fit import make_options, make_test_tensor

SCAN_POINTS = 7
SCAN_RANGE = 2.0
METHODS = ["trust-exact", "tf-trust-exact", "dogleg"]


@pytest.fixture(scope="module")
def tensor():
    with tempfile.TemporaryDirectory() as outdir:
        yield make_test_tensor(outdir)


def _fitted(filename, method, precondition=False):
    indata_obj = inputdata.FitInputData(filename)
    param_model = load_model("Mu", indata_obj)
    opts = dict(minimizerMethod=method, earlyStopping=20)
    if precondition:
        opts.update(precondition=True, preconditionBlocks="none")
    f = fitter.Fitter(indata_obj, param_model, make_options(**opts))
    f.set_nobs(indata_obj.data_obs)
    f.minimize()
    # nll_scan scales its range by sqrt(cov[idx, idx])
    _, grad, hess = f.loss_val_grad_hess()
    _, cov = edmval_cov(grad, hess)
    f.cov.assign(cov)
    return f


@pytest.fixture(scope="module")
def reference(tensor):
    """trust-krylov profiles: its Krylov space never contains a frozen
    coordinate, so its scans are the reference."""
    f = _fitted(tensor, "trust-krylov")
    return {p: f.nll_scan(p, SCAN_RANGE, SCAN_POINTS)[:2] for p in ("sig", "bkg_norm")}


# precondition=True scans bkg_norm, so that the unconstrained POI is
# preconditioned while a constrained nuisance is frozen; frozen parameters
# never enter a preconditioner block, so scanning the POI would leave nothing
# to precondition.
@pytest.mark.parametrize(
    "param,precondition", [("sig", False), ("bkg_norm", False), ("bkg_norm", True)]
)
@pytest.mark.parametrize("method", METHODS)
def test_scan_keeps_the_frozen_parameter(
    tensor, reference, method, param, precondition
):
    f = _fitted(tensor, method, precondition=precondition)
    if precondition:
        # the transform is built per fit leg; check it really has a block here
        f.freeze_params(param)
        assert f._build_preconditioner().enabled
        f.defreeze_params(param)

    scan_vals, dnlls, detail = f.nll_scan(
        param, SCAN_RANGE, SCAN_POINTS, save_detail=True
    )

    idx = int(np.where(f.parms.astype(str) == param)[0][0])
    # exactly where the scan put it, at every point: not close, equal
    np.testing.assert_array_equal(detail["x"][:, idx], scan_vals)

    ref_vals, ref_dnlls = reference[param]
    np.testing.assert_allclose(scan_vals, ref_vals, rtol=0, atol=1e-6)
    np.testing.assert_allclose(dnlls, ref_dnlls, rtol=0, atol=1e-5)
    # a real profile, not the flat curve of a parameter walking back to its
    # best fit (the +-2 sigma points sit at dnll ~ 2)
    assert dnlls[0] > 1.0 and dnlls[-1] > 1.0, dnlls


@pytest.mark.parametrize("method", METHODS)
def test_freeze_parameters_option(tensor, method):
    """--freezeParameters: a fit from a displaced frozen value keeps it."""
    indata_obj = inputdata.FitInputData(tensor)
    param_model = load_model("Mu", indata_obj)
    f = fitter.Fitter(
        indata_obj,
        param_model,
        make_options(
            minimizerMethod=method, earlyStopping=20, freezeParameters=["sig"]
        ),
    )
    f.set_nobs(indata_obj.data_obs)
    idx = int(np.where(f.parms.astype(str) == "sig")[0][0])
    x = f.x.numpy().copy()
    x[idx] = 0.9  # sqrt storage: signal strength 0.81, far from its best fit
    f.x.assign(x)
    f.minimize()
    assert f.x.numpy()[idx] == 0.9
    # and the floating parameters are at their profile minimum
    _, grad = f.loss_val_grad()
    assert np.max(np.abs(grad.numpy()[f.floating_indices])) < 1e-4
