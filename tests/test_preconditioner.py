"""Tests for the optional parameter preconditioning.

The transform is a pure reparameterisation, so the decisive test is that a fit
run with it converges to the *same* parameters, uncertainties and NLL as one
run without it. The unit tests below check the algebra that guarantees this:
T^T H T = I on the block, and the chain rule for the gradient / Hessian-vector
product that the scipy callbacks rely on.
"""

import os
import tempfile

import numpy as np
import pytest

from rabbit import fitter, inputdata
from rabbit.param_models.helpers import load_model
from rabbit.preconditioner import Preconditioner, select_indices

from .test_sparse_fit import check_results, make_options, make_test_tensor


def _spd(n, seed=0, cond=1e6):
    """A symmetric positive definite matrix with a controlled condition number."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eig = np.geomspace(1.0, cond, n)
    return q @ np.diag(eig) @ q.T


# -- unit tests: the algebra ---------------------------------------------


def test_identity_is_exact_noop():
    theta = np.arange(5, dtype=float)
    pc = Preconditioner.identity(theta)
    assert not pc.enabled
    y = pc.from_physical(theta)
    np.testing.assert_allclose(y, np.zeros(5))
    np.testing.assert_allclose(pc.to_physical(y), theta)
    g = np.array([1.0, -2.0, 3.0, 0.5, 0.0])
    np.testing.assert_allclose(pc.grad_to_internal(g), g)
    h = _spd(5)
    np.testing.assert_allclose(pc.hess_to_internal(h), h)


def test_round_trip():
    n = 8
    theta_ref = np.linspace(-1, 1, n)
    h = _spd(n, seed=1)
    idx = np.arange(2, 7)
    pc = Preconditioner.from_hessian(h, theta_ref, idx)
    assert pc.enabled and pc.nblock == idx.size

    rng = np.random.default_rng(3)
    y = rng.normal(size=n)
    np.testing.assert_allclose(pc.from_physical(pc.to_physical(y)), y, atol=1e-10)

    theta = theta_ref + rng.normal(size=n)
    np.testing.assert_allclose(
        pc.to_physical(pc.from_physical(theta)), theta, atol=1e-10
    )


def test_whitens_the_reference_hessian_on_the_block():
    """T^T H T must be the identity on the block -- that is the whole point."""
    n = 10
    h = _spd(n, seed=2, cond=1e8)
    idx = np.arange(n)
    pc = Preconditioner.from_hessian(h, np.zeros(n), idx, ridge=0.0)
    hy = pc.hess_to_internal(h)
    np.testing.assert_allclose(hy, np.eye(n), atol=1e-6)
    # and the conditioning is genuinely improved
    assert np.linalg.cond(hy) < 1e3 < np.linalg.cond(h)


def test_partial_block_leaves_the_rest_untouched():
    """Parameters outside the block pass through, including their Hessian block."""
    n = 6
    h = _spd(n, seed=4)
    idx = np.array([0, 1, 2])
    pc = Preconditioner.from_hessian(h, np.zeros(n), idx, ridge=0.0)
    hy = pc.hess_to_internal(h)
    # block-block whitened
    np.testing.assert_allclose(hy[np.ix_(idx, idx)], np.eye(idx.size), atol=1e-8)
    # outside-outside untouched
    rest = np.array([3, 4, 5])
    np.testing.assert_allclose(
        hy[np.ix_(rest, rest)], h[np.ix_(rest, rest)], atol=1e-12
    )
    # off-diagonal coupling is transformed one-sided, and stays symmetric
    np.testing.assert_allclose(hy, hy.T, atol=1e-8)


def test_gradient_follows_the_chain_rule():
    """grad_y == T^T grad_theta, checked against finite differences."""
    n = 6
    a = _spd(n, seed=5)
    b = np.linspace(-0.5, 0.5, n)
    theta_ref = np.zeros(n)
    pc = Preconditioner.from_hessian(a, theta_ref, np.arange(n), ridge=0.0)

    def loss_theta(theta):
        return 0.5 * theta @ a @ theta + b @ theta

    def loss_y(y):
        return loss_theta(pc.to_physical(y))

    rng = np.random.default_rng(6)
    y0 = rng.normal(size=n) * 0.1
    grad_theta = a @ pc.to_physical(y0) + b
    analytic = pc.grad_to_internal(grad_theta)

    numeric = np.empty(n)
    eps = 1e-6
    for i in range(n):
        yp, ym = y0.copy(), y0.copy()
        yp[i] += eps
        ym[i] -= eps
        numeric[i] = (loss_y(yp) - loss_y(ym)) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_hessp_matches_the_dense_transform():
    """hessp_to_internal must agree with hess_to_internal @ p."""
    n = 7
    a = _spd(n, seed=7)
    pc = Preconditioner.from_hessian(a, np.zeros(n), np.arange(1, 6), ridge=0.0)
    hy = pc.hess_to_internal(a)
    rng = np.random.default_rng(8)
    p = rng.normal(size=n)
    got = pc.hessp_to_internal(p, lambda v: a @ v)
    np.testing.assert_allclose(got, hy @ p, atol=1e-8)


def test_explicit_inverse_and_triangular_fallback_agree():
    """The cached-inverse fast path must match the triangular-solve fallback.

    Both branches are live (the fallback triggers if the inverse cannot be
    formed), so they have to give the same answer.
    """
    n = 9
    h = _spd(n, seed=11, cond=1e8)
    idx = np.arange(1, 8)
    fast = Preconditioner.from_hessian(h, np.zeros(n), idx, ridge=0.0)
    slow = Preconditioner.from_hessian(h, np.zeros(n), idx, ridge=0.0)
    assert fast._linv is not None
    slow._linv = None  # force the triangular-solve path

    rng = np.random.default_rng(12)
    v = rng.normal(size=n)
    np.testing.assert_allclose(fast.to_physical(v), slow.to_physical(v), atol=1e-10)
    np.testing.assert_allclose(
        fast.grad_to_internal(v), slow.grad_to_internal(v), atol=1e-10
    )
    np.testing.assert_allclose(
        fast.hess_to_internal(h), slow.hess_to_internal(h), atol=1e-8
    )
    np.testing.assert_allclose(
        fast.hessp_to_internal(v, lambda x: h @ x),
        slow.hessp_to_internal(v, lambda x: h @ x),
        atol=1e-8,
    )


def test_singular_block_falls_back_to_identity():
    """A preconditioner must never break a fit."""
    n = 5
    h = np.zeros((n, n))  # completely degenerate
    pc = Preconditioner.from_hessian(h, np.zeros(n), np.arange(n))
    assert not pc.enabled


def test_rank_deficient_block_is_ridged_into_shape():
    n = 5
    h = _spd(n, seed=9)
    h[:, -1] = h[:, 0]  # exact linear dependence
    h[-1, :] = h[0, :]
    pc = Preconditioner.from_hessian(h, np.zeros(n), np.arange(n), ridge=1e-8)
    assert pc.enabled


def test_empty_block_falls_back_to_identity():
    pc = Preconditioner.from_hessian(_spd(4), np.zeros(4), np.array([], dtype=int))
    assert not pc.enabled


# -- unit tests: scope selection ----------------------------------------


def test_default_scope_is_the_unconstrained_parameters():
    parms = np.array(["poi", "a", "b", "c"])
    cw = np.array([0.0, 1.0, 0.0, 1.0])
    frozen = np.zeros(4, dtype=bool)
    idx = select_indices(parms, cw, frozen)
    np.testing.assert_array_equal(idx, [0, 2])


def test_frozen_parameters_are_always_excluded():
    parms = np.array(["poi", "a", "b", "c"])
    cw = np.zeros(4)
    frozen = np.array([False, True, False, False])
    idx = select_indices(parms, cw, frozen)
    np.testing.assert_array_equal(idx, [0, 2, 3])


def test_selection_by_regex_and_by_group():
    parms = np.array(["poi", "eff_a", "eff_b", "other"])
    cw = np.ones(4)
    frozen = np.zeros(4, dtype=bool)

    def match_fn(exprs, names):
        import re

        return [n for n in names if any(re.fullmatch(e, n) for e in exprs)]

    idx = select_indices(parms, cw, frozen, expressions=["eff_.*"], match_fn=match_fn)
    np.testing.assert_array_equal(idx, [1, 2])

    idx = select_indices(
        parms,
        cw,
        frozen,
        expressions=["mygroup"],
        match_fn=match_fn,
        groups=["mygroup"],
        group_idxs=[[3]],
    )
    np.testing.assert_array_equal(idx, [3])


# -- the invariance test -------------------------------------------------


def make_polynomial_tensor(outdir, order=6, nbins=30):
    """Background with ``order`` *unconstrained* polynomial shape systematics.

    Mirrors the situation preconditioning is for: a smooth basis that is not
    orthogonal under the data's own weight, so the coefficients are strongly
    correlated and the block is badly conditioned. Plain monomials x^k are used
    deliberately -- the Vandermonde-like Gram matrix is about as ill-conditioned
    as it gets, which is the point.
    """
    import hist

    from rabbit import tensorwriter

    rng = np.random.default_rng(7)
    ax = hist.axis.Regular(nbins, -1, 1, name="x")
    x = ax.centers

    truth = 1000.0 * np.exp(-0.5 * (x / 0.6) ** 2) + 200.0
    h_data = hist.Hist(ax, storage=hist.storage.Double())
    h_data.values()[...] = rng.poisson(truth).astype(float)

    h_bkg = hist.Hist(ax, storage=hist.storage.Weight())
    h_bkg.values()[...] = truth
    h_bkg.variances()[...] = truth

    h_sig = hist.Hist(ax, storage=hist.storage.Weight())
    h_sig.values()[...] = 50.0 * np.exp(-0.5 * (x / 0.2) ** 2)
    h_sig.variances()[...] = h_sig.values()

    writer = tensorwriter.TensorWriter()
    writer.add_channel(h_data.axes, "ch0")
    writer.add_data(h_data, "ch0")
    writer.add_process(h_sig, "sig", "ch0", signal=True)
    writer.add_process(h_bkg, "bkg", "ch0")

    for k in range(1, order + 1):
        w = 0.05 * x**k
        up = h_bkg.copy()
        dn = h_bkg.copy()
        up.values()[...] = h_bkg.values() * (1 + w)
        dn.values()[...] = h_bkg.values() * (1 - w)
        writer.add_systematic(
            [up, dn],
            f"poly{k}",
            "bkg",
            "ch0",
            symmetrize="average",
            constrained=False,
        )

    writer.write(outfolder=outdir, outfilename="test_poly")
    return os.path.join(outdir, "test_poly.hdf5")


def _run(filename, method, **kw):
    """Fit ``filename`` and return the results plus the preconditioner used."""
    indata_obj = inputdata.FitInputData(filename)
    param_model = load_model("Mu", indata_obj)
    options = make_options(minimizerMethod=method, **kw)
    f = fitter.Fitter(indata_obj, param_model, options)
    f.set_nobs(indata_obj.data_obs)
    # built separately purely so the test can assert the block is non-trivial;
    # fit() builds its own at the same point
    pc = f._build_preconditioner()
    f.minimize()
    val, grad, hess = f.loss_val_grad_hess()
    from rabbit.tfhelpers import edmval_cov

    edmval, cov = edmval_cov(grad, hess)
    cov_np = np.asarray(cov.numpy() if hasattr(cov, "numpy") else cov)
    res = dict(
        param=f.x[: param_model.nparams].numpy(),
        theta=f.x[param_model.nparams :].numpy(),
        param_err=np.sqrt(np.diag(cov_np)[: param_model.nparams]),
        nll=f.reduced_nll().numpy(),
        edmval=edmval,
        parms=f.parms,
    )
    return res, pc


@pytest.mark.parametrize("method", ["trust-krylov", "trust-exact"])
def test_fit_is_invariant_under_preconditioning(method):
    """Same minimum, same uncertainties, same NLL -- with trust-krylov kept.

    Covers both the hessp path (trust-krylov) and the dense-hess path
    (trust-exact), since the two transform different callbacks. The scope is
    forced to every parameter so the transform is a genuinely dense one with
    off-diagonal coupling, not a per-parameter rescaling.
    """
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_test_tensor(tmp)
        plain, _ = _run(filename, method)
        pre, pc = _run(filename, method, precondition=True, preconditionParams=[".*"])

        assert pc.enabled and pc.nblock > 1, "preconditioning block must be non-trivial"
        assert check_results("plain", plain, "preconditioned", pre)


@pytest.mark.parametrize("method", ["trust-krylov", "trust-exact"])
def test_fit_is_invariant_on_an_ill_conditioned_block(method):
    """The case this feature exists for: many correlated unconstrained params.

    Uses the default scope (unconstrained parameters), so it also checks that
    the default picks the right block.
    """
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_polynomial_tensor(tmp, order=6)
        plain, _ = _run(filename, method)
        pre, pc = _run(filename, method, precondition=True)

        # default scope must have found the unconstrained polynomial block
        assert pc.enabled and pc.nblock > 1
        # and it must actually be badly conditioned before, well conditioned after
        assert pc.cond_before > 1e3
        assert pc.cond_after < 1e2
        assert check_results("plain", plain, "preconditioned", pre)
