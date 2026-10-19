"""Limited-MC-statistics de-biasing: what is reported from a de-biased fit.

Goodness of fit, impacts, adjusted likelihood scans, the saturated projection
test, rabbit_limit.py, the meat of every objective term, refused option
combinations and the coverage warnings.
"""

import importlib.util

import hist
import numpy as np
import pytest
import scipy.stats
import tensorflow as tf

from rabbit import fitter
from tests.mcstat_helpers import (
    LINEAR_CHISQ,
    REPO,
    SMALL_NBINS,
    fit_toy,
    make_small_fold_toy,
)

# ---------------------------------------------------------------------------
# adjusted profile-likelihood scans
# ---------------------------------------------------------------------------


def test_adjusted_scans_follow_the_sandwich(tmp_path):
    """For a quadratic (linear chisq) de-biased objective the profiled
    statistic is d^T C^-1 d exactly; the adjusted statistic must then be the
    sandwich Wald form d^T Sigma^-1 d, in 1D and 2D, and the adjusted contour
    interval must be +-sigma_sandwich."""
    fn, _ = make_small_fold_toy(tmp_path, 2)
    f = fit_toy(fn, tmp_path, *LINEAR_CHISQ, "--mcStatDebias", "kfold")
    _, _, h = f.loss_val_grad_hess()
    C = np.linalg.inv(np.asarray(h))
    S = np.asarray(f.cov_twohalf_sandwich(h))
    assert not np.allclose(C, S)
    f.cov.assign(S)
    f.set_mcstat_scan_covariances(np.arange(2), C, S)
    assert f.mcstat_scan_needs_adjustment()
    xh = f.x.numpy().copy()

    xs, dnll, _ = f.nll_scan("flat", 2.0, 5)
    W = 2 * dnll * f.mcstat_scan_adjustment([0], (xs - xh[0])[:, None])
    np.testing.assert_allclose(W, (xs - xh[0]) ** 2 / S[0, 0], rtol=1e-6, atol=1e-9)

    xs2, ys2, dnll2 = f.nll_scan2D(("flat", "step"), 1.5, 5)
    gx, gy = np.meshgrid(
        np.asarray(xs2) - xh[0], np.asarray(ys2) - xh[1], indexing="ij"
    )
    d = np.stack([gx.ravel(), gy.ravel()], -1)
    W2 = 2 * dnll2.ravel() * f.mcstat_scan_adjustment([0, 1], d)
    wald = np.einsum("ni,ij,nj->n", d, np.linalg.inv(S), d)
    np.testing.assert_allclose(W2, wald, rtol=1e-6, atol=1e-9)

    q = 1.0 / float(f.mcstat_scan_adjustment([0], [[1.0]])[0])
    interval, _ = f.contour_scan("flat", float(f.reduced_nll().numpy()), q)
    np.testing.assert_allclose(np.abs(interval), np.sqrt(S[0, 0]), rtol=1e-4)


# ---------------------------------------------------------------------------
# goodness of fit at the de-biased minimum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3])
def test_gof_statistic_and_reference(k, tmp_path):
    """The goodness-of-fit statistic is 2 * reduced_nll() of the de-biased
    objective at its minimum, bin-by-bin stat included, as for an ordinary fit.
    For a chisq data term without bin-by-bin stat it is exactly the cross-fit
    chi2 sum_b (y-n_A)(y-n_B)/V, for any k, and its reference mean is
    nbins - p (the MC fluctuations cancel). With bin-by-bin stat the profiled
    beta leaves the fraction s_b ~ 1/(1+rho_b) of a residual, and the reference
    mean becomes sum_b s_b - p (each jackknife term separately profiled)."""
    fn, arr = make_small_fold_toy(tmp_path, k)
    y = arr["data"]
    a, b = k // 2, k - k // 2
    Ti = [np.stack([arr["folds"][0][i], arr["folds"][1][i]], 1) for i in range(k)]
    TA, TB = (k / a) * sum(Ti[:a]), (k / b) * sum(Ti[a:])

    f = fit_toy(fn, tmp_path, *LINEAR_CHISQ, "--mcStatDebias", "kfold")
    x = f.x.numpy()
    q = 2 * float(f.reduced_nll().numpy())
    np.testing.assert_allclose(q, np.sum((y - TA @ x) * (y - TB @ x) / y), rtol=1e-8)
    shrink, rho = (np.asarray(v) for v in f._mcstat_gof_bin_inputs())
    np.testing.assert_allclose(shrink, 1.0)
    g = f.mcstat_gof(q, trace_AinvH=2.0)
    assert g["mean"] == pytest.approx(SMALL_NBINS - 2)
    assert 0.0 <= g["pvalue"] <= 1.0

    fb = fit_toy(
        fn, tmp_path, "--chisqFit", "--allowNegativeParam", "--mcStatDebias", "kfold"
    )
    shrink_b, rho_b = (np.asarray(v) for v in fb._mcstat_gof_bin_inputs())
    np.testing.assert_allclose(shrink_b, 1.0 / (1.0 + rho_b), rtol=0.05)
    gb = fb.mcstat_gof(2 * float(fb.reduced_nll().numpy()), trace_AinvH=2.0)
    np.testing.assert_allclose(gb["mean"], np.sum(shrink_b) - 2.0, rtol=1e-8)
    assert 0.0 <= gb["pvalue"] <= 1.0


# ---------------------------------------------------------------------------
# impacts: mcStatDebias and the combined mcStat group
# ---------------------------------------------------------------------------


def _debiased_postfit(tmp_path, *extra):
    fn, _ = make_small_fold_toy(tmp_path, 4)
    f = fit_toy(fn, tmp_path, "--allowNegativeParam", "--mcStatDebias", "kfold", *extra)
    _, g, h = f.loss_val_grad_hess()
    _, bread, C, S = f.mcstat_postfit_covariances(g, h)
    f.cov.assign(S)
    f.mcstat_cov_curv.assign(C)
    return f, bread, np.asarray(C), np.asarray(S)


def test_impact_groups_add_up_to_the_reported_uncertainty(tmp_path):
    """The traditional and global impacts of a de-biased fit decompose its
    curvature covariance and append mcStatDebias (reported - curvature) and the
    combined mcStat (bin-by-bin stat and de-biasing in quadrature), so the
    groups add up to the reported (sandwich) uncertainty; the same holds for the
    histogram global impacts."""
    from rabbit.mappings import helpers as mh

    f, bread, C, S = _debiased_postfit(tmp_path)
    extra = tf.constant(np.diag(S) - np.diag(C))
    sigma = np.sqrt(np.diag(S))[:2]
    for _, grouped in (
        f.impacts_parms(bread, cov=tf.constant(C), mcstat_extra_var=extra),
        f.global_impacts_parms(cov=tf.constant(C), mcstat_extra_var=extra),
    ):
        g = np.asarray(grouped)  # [stat, binByBinStat, mcStatDebias, mcStat]
        assert g.shape[1] == 4
        np.testing.assert_allclose(np.sqrt(np.sum(g[:, :3] ** 2, axis=1)), sigma)
        np.testing.assert_allclose(g[:, 3], np.hypot(g[:, 1], g[:, 2]))
        assert np.all(g[:, 2] > 0)

    mapping = mh.load_mapping("BaseMapping", f.indata)
    exp, aux = f.expected_events(mapping, compute_global_impacts=True)
    var, grouped = np.asarray(aux[0]).ravel(), np.asarray(aux[3])
    grouped = grouped.reshape(-1, grouped.shape[-1])
    np.testing.assert_allclose(np.sum(grouped[:, :3] ** 2, axis=1), var, rtol=1e-8)
    np.testing.assert_allclose(grouped[:, 3], np.hypot(grouped[:, 1], grouped[:, 2]))


# ---------------------------------------------------------------------------
# saturated projection test of a de-biased fit
# ---------------------------------------------------------------------------


def test_weighted_chi2_sf():
    rng = np.random.default_rng(1)
    np.testing.assert_allclose(
        [fitter.weighted_chi2_sf(x, [1.0] * 5) for x in (1, 5, 12)],
        scipy.stats.chi2.sf([1, 5, 12], 5),
    )
    for w in ([0.5, 1.3, 2.0, 0.2], [3.0, 0.3], list(rng.uniform(0.5, 2, 30))):
        s = (rng.standard_normal((200000, len(w))) ** 2 * np.asarray(w)).sum(1)
        xs = np.quantile(s, [0.3, 0.9, 0.99])
        np.testing.assert_allclose(
            [fitter.weighted_chi2_sf(x, w) for x in xs], [0.7, 0.1, 0.01], atol=4e-3
        )


def test_projection_test_reference(tmp_path):
    """The reference of the de-biased saturated projection test,
    sum_j mu_j chi2_1 with mu = eig(A~^+ B~) of the bin scales' profiled
    curvature and score variance at the null point. For an ordinary fit (bread
    = meat) every mu is 1, and a bin-scale direction degenerate with the
    analysis parameters (here the common normalization, as both processes are
    free) is not testable and drops out: npou - 1 values."""
    import copy

    from rabbit.param_models.param_model import (
        CompositeParamModel,
        SaturatedProjectModel,
    )

    ngroup = 6
    groups = np.repeat(np.arange(ngroup), SMALL_NBINS // ngroup)
    chinfo = {"ch0": {"axes": [hist.axis.Regular(ngroup, 0, 1, name="g")]}}
    fn, _ = make_small_fold_toy(tmp_path, 2)
    for debias, n_mu in (("none", ngroup - 1), ("kfold", ngroup - 1)):
        f = fit_toy(fn, tmp_path, "--allowNegativeParam", "--mcStatDebias", debias)
        orig = f.param_model
        comp = CompositeParamModel(
            [orig, SaturatedProjectModel(f.indata, chinfo, {"ch0": groups})]
        )
        fs = copy.deepcopy(f)
        fs.init_fit_parms(comp, [], unblind=[], blinding_group=[], freeze_parameters=[])
        fs.xdefaultassign()
        xm = f.x.numpy()
        fs.x[: orig.nparams].assign(xm[: orig.nparams])
        fs.x[comp.nparams :].assign(xm[orig.nparams :])
        fs.arm_regularizers()
        psi = np.arange(orig.nparams, comp.nparams)
        if debias != "none":
            assert fs.mcstat_projection_check(psi)[0]
        if debias == "none":
            _, _, h = fs.loss_val_grad_hess()
            A0 = H0 = np.asarray(h)
        else:
            A0, H0 = fs.mcstat_curvature_and_meat()
        fs.minimize(mcstat_warm_start=False)
        dq = 2 * float(f.reduced_nll().numpy() - fs.reduced_nll().numpy())
        assert dq >= -1e-6
        p, mu = fs.mcstat_projection_test(dq, psi, A0, H0)
        assert len(mu) == n_mu
        assert np.all(mu > 0)
        if debias == "none":
            np.testing.assert_allclose(mu, 1.0, rtol=1e-6)
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# rabbit_limit.py
# ---------------------------------------------------------------------------


def test_rabbit_limit_uses_the_sandwich_and_adjusted_contours(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "_rabbit_limit_for_mcstat_tests", REPO / "bin" / "rabbit_limit.py"
    )
    rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rl)
    fn, _ = make_small_fold_toy(tmp_path, 4)
    f = fit_toy(fn, tmp_path, "--allowNegativeParam", "--mcStatDebias", "kfold")
    rl._postfit_cov(f)
    _, g, h = f.loss_val_grad_hess()
    C = np.linalg.inv(np.asarray(h))
    S = np.asarray(f.cov_twohalf_sandwich(h))
    np.testing.assert_allclose(f.cov.numpy(), S, rtol=1e-8)
    np.testing.assert_allclose(
        rl._mcstat_contour_factor(f, idx=0), C[0, 0] / S[0, 0], rtol=1e-8
    )


def test_projection_test_refuses_an_ill_posed_projection(tmp_path):
    """With one bin scale per bin, the scales absorb any change of process
    composition in the full prediction, which the half predictions still feel:
    a direction the data cannot constrain but the de-biased objective is not
    flat along. The check must flag it (the saturated de-biased fit would
    follow template noise)."""
    import copy

    from rabbit.param_models.param_model import (
        CompositeParamModel,
        SaturatedProjectModel,
    )

    fn, _ = make_small_fold_toy(tmp_path, 2)
    f = fit_toy(fn, tmp_path, "--allowNegativeParam", "--mcStatDebias", "kfold")
    chinfo = {"ch0": {"axes": [hist.axis.Regular(SMALL_NBINS, 0, 1, name="x")]}}
    orig = f.param_model
    comp = CompositeParamModel(
        [
            orig,
            SaturatedProjectModel(f.indata, chinfo, {"ch0": np.arange(SMALL_NBINS)}),
        ]
    )
    fs = copy.deepcopy(f)
    fs.init_fit_parms(comp, [], unblind=[], blinding_group=[], freeze_parameters=[])
    fs.xdefaultassign()
    xm = f.x.numpy()
    fs.x[: orig.nparams].assign(xm[: orig.nparams])
    fs.x[comp.nparams :].assign(xm[orig.nparams :])
    fs.arm_regularizers()
    ok, worst = fs.mcstat_projection_check(np.arange(orig.nparams, comp.nparams))
    assert not ok and worst > 1e-3


# ---------------------------------------------------------------------------
# review follow-ups: regularization in the meat, refused combinations
# ---------------------------------------------------------------------------


def test_meat_includes_every_term_of_the_ordinary_objective(tmp_path):
    """The sandwich meat is the curvature of the ordinary (--mcStatDebias none)
    objective at the de-biased point, regularization included: with a
    regularizer it must equal the Hessian of that objective, plus that of the
    bin-by-bin stat compensator (marginal instead of profiled beta; here the
    small total-yield term of a lite multiplicative beta), plus the
    split-contrast meat (tested against its closed form separately)."""
    from rabbit.regularization.regularizer import Regularizer

    class Penalty(Regularizer):
        needs_observables = False

        def __init__(self):
            self._idx = None

        def set_expectations(self, initial_params, initial_observables, parms=None):
            self._idx = self.resolve_indices(parms, ["flat"], who="Penalty")["flat"]

        def compute_nll_penalty(self, params, observables):
            return 50.0 * tf.square(params[self._idx] - 1.0)

    fn, _ = make_small_fold_toy(tmp_path, 2)
    f = fit_toy(fn, tmp_path, "--allowNegativeParam", "--mcStatDebias", "kfold")
    f.tau.assign(1.0)
    f.regularizers = [Penalty()]
    f.arm_regularizers()
    meat = np.asarray(f.mcstat_meat("observed"))
    with f._ordinary_likelihood():
        _, _, h_ordinary = f.loss_val_grad_hess()
    h_comp = f._scalar_hessian(
        "test", lambda: f._compute_mcstat_bb_compensator(meat=True)
    )
    assert np.abs(np.asarray(h_comp)).max() > 0
    B_split = np.asarray(f.mcstat_split_meat())
    np.testing.assert_allclose(
        meat, np.asarray(h_ordinary) + np.asarray(h_comp) + B_split, rtol=1e-8
    )


def test_data_propagated_sandwich_refused_for_fold_methods(tmp_path):
    """cov_dataprop_sandwich differentiates the full templates only, while a
    fold objective depends on the fold templates: refused."""
    fn, _ = make_small_fold_toy(tmp_path, 2)
    with pytest.raises(NotImplementedError, match="only available for"):
        fit_toy(
            fn,
            tmp_path,
            "--allowNegativeParam",
            "--mcStatDebias",
            "kfold",
            "--mcStatDebiasCov",
            "dataPropagated",
        )


@pytest.mark.parametrize(
    "args,expect",
    [
        (["--mcStatDebias", "kfold", "--noBinByBinStat"], "without bin-by-bin stat"),
        (
            [
                "--mcStatDebias",
                "kfold",
                "--covMode",
                "fisher",
                "--binByBinStatType",
                "gamma",
                "--binByBinStatDynamic",
            ],
            "--covMode fisher with --binByBinStatType gamma",
        ),
        (["--mcStatDebias", "kfold", "--mcStatDebiasCov", "curvature"], "curvature"),
        (
            [
                "--mcStatDebias",
                "kfold",
                "--binByBinStatType",
                "gamma",
                "--binByBinStatDynamic",
            ],
            None,
        ),
        (["--mcStatDebias", "kfold", "--covMode", "fisher"], None),
        (["--mcStatDebias", "none", "--noBinByBinStat"], None),
    ],
    ids=[
        "folds-nobbb",
        "fisher-gamma-dyn",
        "curvature",
        "observed-gamma-dyn",
        "fisher-gamma-lite",
        "none",
    ],
)
def test_coverage_warnings_for_the_known_mis_covering_cases(args, expect, tmp_path):
    """The configurations whose reported covariance is known to under- or
    over-cover (docs/mcstat/RESULTS.md §9p, §9u) warn; the others do not."""
    fn, _ = make_small_fold_toy(tmp_path, 2)
    f = fit_toy(fn, tmp_path, "--allowNegativeParam", *args)
    msgs = f.mcstat_coverage_warnings()
    if expect is None:
        assert msgs == []
    else:
        assert len(msgs) == 1 and expect in msgs[0]
