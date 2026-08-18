"""Tests for restarting the minimizer after an early-stopping stall.

scipy's trust-region loop shrinks the trust radius by 4x on every rejected step
with no lower bound, and holds the radius in a local variable. Once it has
collapsed the method takes infinitesimal steps and the loss stops changing far
from any minimum; a fresh minimize() call resets the radius and the descent
resumes. These tests cover the plumbing that turns that stall into a restart
instead of a give-up.
"""

import tempfile
from types import SimpleNamespace

import numpy as np

from rabbit import fitter, inputdata
from rabbit.fitter import FitterCallback, _merge_callbacks
from rabbit.param_models.helpers import load_model

from .test_preconditioner import make_polynomial_tensor
from .test_sparse_fit import check_results, make_options, make_test_tensor


def _result(fun, x):
    return SimpleNamespace(fun=fun, x=np.asarray(x, dtype=float))


def test_stopped_early_flag_set_only_on_a_stall():
    """fit() keys the restart off this flag, so it must not fire otherwise."""
    cb = FitterCallback(np.zeros(2), early_stopping=3)
    # a steadily improving fit never stalls
    for i, loss in enumerate([10.0, 9.0, 8.0, 7.0, 6.0, 5.0]):
        cb(_result(loss, [i, i]))
    assert not cb.stopped_early

    # a flat one does
    cb = FitterCallback(np.zeros(2), early_stopping=3)
    raised = False
    try:
        for loss in [10.0, 9.0, 9.0, 9.0, 9.0, 9.0]:
            cb(_result(loss, [0, 0]))
    except ValueError:
        raised = True
    assert raised and cb.stopped_early


def test_disabled_early_stopping_never_stalls():
    cb = FitterCallback(np.zeros(2), early_stopping=-1)
    for _ in range(20):
        cb(_result(5.0, [0, 0]))
    assert not cb.stopped_early


def test_merge_callbacks_concatenates_into_one_continuous_run():
    """Callers report on the whole fit, so restarts must look continuous."""
    a = FitterCallback(np.zeros(2), early_stopping=-1)
    a.loss_history = [10.0, 9.0]
    a.time_history = [1.0, 2.0]
    a.iiter = 2

    b = FitterCallback(np.ones(2), early_stopping=-1)
    b.loss_history = [8.0, 7.0]
    b.time_history = [0.5, 1.5]  # clocked from its own construction
    b.iiter = 2
    b.xval = np.array([3.0, 4.0])

    merged = _merge_callbacks(a, b)
    assert merged is a
    assert merged.loss_history == [10.0, 9.0, 8.0, 7.0]
    # second run's times offset by the first run's elapsed
    assert merged.time_history == [1.0, 2.0, 2.5, 3.5]
    assert merged.iiter == 4
    np.testing.assert_array_equal(merged.xval, [3.0, 4.0])


def test_merge_callbacks_with_no_accumulator_returns_the_first():
    cb = FitterCallback(np.zeros(2), early_stopping=-1)
    assert _merge_callbacks(None, cb) is cb


def _fit(filename, **kw):
    indata_obj = inputdata.FitInputData(filename)
    param_model = load_model("Mu", indata_obj)
    options = make_options(**kw)
    f = fitter.Fitter(indata_obj, param_model, options)
    f.set_nobs(indata_obj.data_obs)
    f.minimize()
    val, grad, hess = f.loss_val_grad_hess()
    from rabbit.tfhelpers import edmval_cov

    edmval, cov = edmval_cov(grad, hess)
    cov_np = np.asarray(cov.numpy() if hasattr(cov, "numpy") else cov)
    return dict(
        param=f.x[: param_model.nparams].numpy(),
        theta=f.x[param_model.nparams :].numpy(),
        param_err=np.sqrt(np.diag(cov_np)[: param_model.nparams]),
        nll=f.reduced_nll().numpy(),
        edmval=edmval,
        parms=f.parms,
    )


def test_restarts_do_not_change_a_fit_that_does_not_stall():
    """--maxRestarts must be inert when nothing stalls."""
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_test_tensor(tmp)
        plain = _fit(filename, maxRestarts=0)
        with_restarts = _fit(filename, earlyStopping=20, maxRestarts=5)
        assert check_results("no restarts", plain, "maxRestarts=5", with_restarts)


def test_restarting_is_on_by_default_and_unbounded():
    """Default is -1: keep restarting while the loss keeps dropping."""

    with tempfile.TemporaryDirectory() as tmp:
        filename = make_test_tensor(tmp)
        indata_obj = inputdata.FitInputData(filename)
        param_model = load_model("Mu", indata_obj)
        # options object without the attribute at all -> the getattr default
        options = make_options()
        del options.maxRestarts
        f = fitter.Fitter(indata_obj, param_model, options)
        assert f.max_restarts == -1


def test_unbounded_restarts_stop_when_a_restart_stops_improving():
    """The improvement check, not a counter, is what ends the loop."""
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_polynomial_tensor(tmp, order=6)
        # unbounded restarts must still terminate
        res = _fit(filename, earlyStopping=10, maxRestarts=-1)
        assert np.isfinite(res["nll"])


def test_restarts_reach_at_least_as_low_a_loss_on_a_hard_model():
    """On a model with many correlated unconstrained params, restarting can
    only help: the restart is only taken after a stall, and is abandoned as
    soon as it stops reducing the loss."""
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_polynomial_tensor(tmp, order=6)
        stop_only = _fit(filename, earlyStopping=10, maxRestarts=0)
        restarted = _fit(filename, earlyStopping=10, maxRestarts=5)
        assert restarted["nll"] <= stop_only["nll"] + 1e-6
