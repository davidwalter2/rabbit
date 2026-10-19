"""Limited-MC-statistics de-biasing: continuous-M (add_mc_stat_moment).

* With a LINEAR POI the point and covariance equal the closed form (H - M)^-1,
  inflating sigma; with rabbit's default x^2 POI the correction is cancelled
  (docs/mcstat/RESULTS.md §9a).
* Composition with every bin-by-bin stat mode, and the variance= convention.
* A small fixed-seed template-fluctuating coverage ensemble
  (tests/mcstat_coverage.py) shows the de-bias direction.
"""

import numpy as np
import pytest

from rabbit.tensorwriter import TensorWriter
from tests.mcstat_helpers import (
    _fisher,
    fit_toy,
    make_mcstat_moment_toy,
    sigma,
)

# ---------------------------------------------------------------------------
# continuous-M (add_mc_stat_moment)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def moment_toys(tmp_path_factory):
    path = tmp_path_factory.mktemp("mcstat_moment")
    fn_noM, _ = make_mcstat_moment_toy(path, with_M=False)
    fn_M, arr = make_mcstat_moment_toy(path, with_M=True)
    return dict(noM=fn_noM, M=fn_M, arrays=arr, out=path)


def _continuous_m_fit(filename, outdir, linear, debias):
    """Fit with --mcStatDebias `debias` and return (rnorm, curvature cov,
    sandwich cov or None) in rnorm space."""
    args = ("--chisqFit", "--noBinByBinStat", "--mcStatDebias", debias)
    if linear:
        args += ("--allowNegativeParam",)
    f = fit_toy(filename, outdir, *args)
    _, grad, hess = f.loss_val_grad_hess()
    _, cov = f.edmval_cov(grad, hess)
    sand = None
    if f.mcstat_M is not None:
        sand = np.asarray(f.cov_mcstat_sandwich(hess))  # A^-1 + A^-1 M A^-1
    x = f.x.numpy()
    jac = np.diag(np.ones_like(x) if linear else 2.0 * x)  # d rnorm / d x
    rnorm = x if linear else x**2
    return (
        rnorm,
        jac @ np.asarray(cov) @ jac,
        None if sand is None else jac @ sand @ jac,
    )


def test_continuous_m_linear_poi_debiases(moment_toys):
    """Linear POI: the -1/2 theta^T M theta penalty gives the closed-form
    de-biased point (H - M)^-1 b, curvature (H - M)^-1 and sandwich
    A^-1 H A^-1, inflating sigma(dif) by > 20% (the scripts' DE-BIASED verdict)."""
    out = moment_toys["out"]
    r_no, C_no, S_no = _continuous_m_fit(moment_toys["noM"], out, True, "none")
    r_M, C_M, S_M = _continuous_m_fit(moment_toys["M"], out, True, "continuousM")
    assert S_no is None and S_M is not None

    arr = moment_toys["arrays"]
    T, V, d = arr["sumw"].T, arr["data"], arr["data"]
    H = _fisher(T, V)
    b = np.einsum("bi,b,b->i", T, 1.0 / V, d)
    A = H - arr["M"]
    Ainv = np.linalg.inv(A)
    np.testing.assert_allclose(r_no, np.linalg.solve(H, b), atol=1e-3)
    np.testing.assert_allclose(r_M, np.linalg.solve(A, b), atol=1e-3)
    assert sigma(C_no) == pytest.approx(sigma(np.linalg.inv(H)), abs=1e-3)
    assert sigma(C_M) == pytest.approx(sigma(Ainv), abs=1e-3)
    assert sigma(S_M) == pytest.approx(sigma(Ainv @ H @ Ainv), abs=1e-3)

    s_no, s_M, s_sand = sigma(C_no), sigma(C_M), sigma(S_M)
    assert s_M > 1.2 * s_no, f"M should inflate sigma(dif): {s_no} -> {s_M}"
    assert s_sand >= s_M, f"sandwich {s_sand} below curvature {s_M}"


def test_continuous_m_squared_poi_is_cancelled(moment_toys):
    """Default x^2 POI transform: the M correction is cancelled by Fisher growth
    at the shifted minimum (docs/mcstat/RESULTS.md §9a), so the curvature sigma(dif) does NOT
    inflate (the scripts' 'no de-bias (cancelled)' verdict). The sandwich still
    adds the A^-1 M A^-1 margin on top of the curvature."""
    out = moment_toys["out"]
    r_no, C_no, _ = _continuous_m_fit(moment_toys["noM"], out, False, "none")
    r_M, C_M, S_M = _continuous_m_fit(moment_toys["M"], out, False, "continuousM")
    s_no, s_M = sigma(C_no), sigma(C_M)
    assert s_M < 1.2 * s_no, f"x^2 POI should cancel the M de-bias: {s_no} -> {s_M}"
    np.testing.assert_allclose(r_M, r_no, atol=1e-3)
    assert sigma(S_M) >= s_M


# ---------------------------------------------------------------------------
# composition with bin-by-bin stat (BB-lite)
# ---------------------------------------------------------------------------


def test_continuous_m_with_bblite_converges_pd(tmp_path):
    """continuous-M + BB-lite with M built on the BB-lite-inflated variance
    (data + sum sumw2): the fit converges (|grad| < 1e-3) to a positive-definite
    Hessian."""
    fn, _ = make_mcstat_moment_toy(tmp_path, with_M=True, bbb_variance=True)
    f = fit_toy(
        fn,
        tmp_path,
        "--chisqFit",
        "--allowNegativeParam",
        "--mcStatDebias",
        "continuousM",
    )
    _, grad, hess = f.loss_val_grad_hess()
    eig = np.linalg.eigvalsh(hess.numpy())
    gmax = float(np.max(np.abs(grad.numpy())))
    assert np.all(eig > 0), f"Hessian not PD: eigenvalues {eig}"
    assert gmax < 1e-3, f"not converged: max|grad| = {gmax}"
    S = np.asarray(f.cov_mcstat_sandwich(hess))
    assert np.all(np.isfinite(S)) and np.all(np.diag(S) > 0)


# ---------------------------------------------------------------------------
# template-fluctuating coverage ensemble (tests/mcstat_coverage.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def coverage_harness():
    from tests import mcstat_coverage

    # the assertions below are calibrated on the harness's default toy
    if (mcstat_coverage.NB, mcstat_coverage.HI) != (200, 5112.0) or list(
        mcstat_coverage.RTRUE
    ) != [1.3, 0.7]:
        pytest.skip("NB/HI/R0/R1 overridden in the environment")
    return mcstat_coverage


def _check_coverage_result(r, ntoy):
    assert r["n"] + r["nbad"] == ntoy
    assert r["nbad"] <= ntoy // 10, f"{r['nbad']} of {ntoy} toy fits failed"
    for key in ("cov_dif", "cov_sum"):
        assert np.isfinite(r[key]) and 0.0 <= r[key] <= 1.0, (key, r[key])
    for key in ("bias_dif", "mean_res", "rms_res", "med_sig"):
        assert np.isfinite(r[key]), (key, r[key])
    assert r["med_sig"] > 0 and r["rms_res"] > 0


# 20 toys per method instead of the standalone tool's 50 to keep CI fast (~1 s per
# toy fit). With the fixed seed the result is deterministic; the statistical margin
# of the bias check is the smallest (the de-biased median residual has a scatter of
# ~0.1 at this size against the 0.4*|naive bias| ~ 0.11 threshold), so run
# tests/mcstat_coverage.py with a larger ensemble when changing the method.
COVERAGE_NTOY = 20
COVERAGE_SEED = 11


def test_coverage_continuous_m_corrects_naive(coverage_harness, tmp_path):
    """Template+data-fluctuating ensemble on the near-degenerate toy with a
    nonzero degenerate-direction truth (rnorm_true = [1.3, 0.7]):

    1. the naive fit is attenuated (median dif residual < -0.10);
    2. continuous-M removes the central-value bias (|bias| < 0.4 |naive bias|);
    3. its sandwich sigma(dif) is > 1.3x the naive one;
    4. it improves the dif coverage by > 0.10.

    Asserts the de-bias DIRECTION, not exact 0.683 coverage: the raw sandwich
    still underestimates the de-biased point's scatter (docs/mcstat/RESULTS.md §9n).
    """
    mc = coverage_harness
    kw = dict(ntoy=COVERAGE_NTOY, seed=COVERAGE_SEED, outdir=str(tmp_path))
    none = mc.run_coverage(debias="none", **kw)
    cm = mc.run_coverage(debias="continuousM", debiasCov="sandwich", **kw)
    _check_coverage_result(none, COVERAGE_NTOY)
    _check_coverage_result(cm, COVERAGE_NTOY)

    assert none["bias_dif"] < -0.10, f"naive should be attenuated: {none}"
    assert abs(cm["bias_dif"]) < 0.4 * abs(
        none["bias_dif"]
    ), f"de-bias should remove the bias: {none['bias_dif']} -> {cm['bias_dif']}"
    assert (
        cm["med_sig"] > 1.3 * none["med_sig"]
    ), f"de-bias should inflate sigma: {none['med_sig']} -> {cm['med_sig']}"
    assert (
        cm["cov_dif"] > none["cov_dif"] + 0.10
    ), f"de-bias should improve coverage: {none['cov_dif']} -> {cm['cov_dif']}"


@pytest.mark.parametrize(
    "config",
    [
        dict(debias="kfold"),
        dict(debias="kfold", covMode="fisher", k=4),
        dict(debias="continuousM", bbb=True),
    ],
    ids=["kfold", "kfold4-fisher", "continuousM-bblite"],
)
def test_coverage_harness_smoke(coverage_harness, tmp_path, config):
    """The harness runs the fold / BB-lite configurations end to end and returns
    finite coverage numbers in [0, 1] (tiny ensemble: smoke test only)."""
    ntoy = 3
    r = coverage_harness.run_coverage(
        ntoy=ntoy, seed=COVERAGE_SEED, outdir=str(tmp_path), **config
    )
    _check_coverage_result(r, ntoy)


# ---------------------------------------------------------------------------
# variance= convention, composition with every bin-by-bin stat mode
# ---------------------------------------------------------------------------


def _moment_fit(fn, tmp_path, *args):
    return fit_toy(fn, tmp_path, "--chisqFit", "--allowNegativeParam", *args)


def test_continuous_m_variance_convention_enforced(tmp_path):
    """M's denominator depends on whether the fit uses bin-by-bin stat, so the
    convention is recorded with the term and a mismatch is refused."""
    fn_bare, _ = make_mcstat_moment_toy(tmp_path, bbb_variance=False)
    fn_bb, _ = make_mcstat_moment_toy(tmp_path, bbb_variance=True)
    _moment_fit(fn_bare, tmp_path, "--noBinByBinStat", "--mcStatDebias", "continuousM")
    _moment_fit(fn_bb, tmp_path, "--mcStatDebias", "continuousM")
    with pytest.raises(RuntimeError, match="'bare' per-bin variance"):
        _moment_fit(fn_bare, tmp_path, "--mcStatDebias", "continuousM")
    with pytest.raises(RuntimeError, match="'bb_inflated' per-bin variance"):
        _moment_fit(
            fn_bb, tmp_path, "--noBinByBinStat", "--mcStatDebias", "continuousM"
        )
    with pytest.raises(ValueError, match="variance must be one of"):
        TensorWriter().add_mc_stat_moment(np.eye(1), ["a"], variance="inflated")


def test_continuous_m_ignored_without_option(moment_toys, tmp_path):
    """A file carrying M fits the plain NLL unless --mcStatDebias continuousM."""
    f0 = _moment_fit(moment_toys["noM"], tmp_path, "--noBinByBinStat")
    f1 = _moment_fit(moment_toys["M"], tmp_path, "--noBinByBinStat")
    assert f1.mcstat_M is None and not f1.mcstat_debias_active
    np.testing.assert_allclose(f1.x.numpy(), f0.x.numpy(), rtol=1e-10)


@pytest.mark.parametrize(
    "bbb,models",
    [
        (["--binByBinStatType", "gamma"], "total"),
        (["--binByBinStatType", "normal-additive"], "none"),
        (
            [
                "--binByBinStatType",
                "normal-multiplicative",
                "--binByBinStatMode",
                "full",
            ],
            "per-process",
        ),
        (
            ["--binByBinStatType", "normal-additive", "--binByBinStatMode", "full"],
            "none",
        ),
        (["--binByBinStatType", "gamma", "--binByBinStatDynamic"], "per-process"),
        (
            ["--binByBinStatType", "normal-additive", "--binByBinStatDynamic"],
            "per-process",
        ),
    ],
    ids=["gamma-lite", "nadd-lite", "nmult-full", "nadd-full", "gamma-dyn", "nadd-dyn"],
)
def test_continuous_m_removes_only_what_bin_by_bin_stat_does_not(bbb, models, tmp_path):
    """Continuous-M with every bin-by-bin stat type and mode. Where the
    bin-by-bin stat variance V(x) = D + sum_c u_c(x)^2 moves with the
    parameters, the profiled likelihood already removes part of M; the
    compensator 1/2 sum_b log(V_b(x) / V_ref,b) (marginal instead of profiled
    beta) restores it, with Hessian M_BB = sum du du^T / V at the reference.
    On the unit-weight toy (normalization POIs, D = data): per-process betas
    model all of M (u_p = sqrt(sumw_p) x_p); a lite multiplicative beta the
    noise of the total yield (u = n(x) / sqrt(sumw)); a static additive one
    none."""
    fn, arr = make_mcstat_moment_toy(tmp_path, bbb_variance=True)
    args = ["--chisqFit", "--allowNegativeParam", *bbb]
    f = fit_toy(fn, tmp_path, *args, "--mcStatDebias", "continuousM")
    sumw = arr["sumw"]
    tot = sumw.sum(0)
    x = f.x.numpy()[:2]
    if models == "none":
        assert f.mcstat_bb_vref is None
    else:
        # Hessian of 1/2 log V = 1/2 V'' / V - 1/2 V' V'^T / V^2 at x
        if models == "per-process":
            V = arr["data"] + x**2 @ sumw
            dV = 2 * sumw * x[:, None]
            d2V = 2 * np.einsum("ib,ij->ijb", sumw, np.eye(2))
        else:
            nx = x @ sumw
            V = arr["data"] + nx**2 / tot
            dV = 2 * nx * sumw / tot
            d2V = 2 * np.einsum("ib,jb->ijb", sumw, sumw) / tot
        ref = 0.5 * np.sum(d2V / V - np.einsum("ib,jb->ijb", dV, dV) / V**2, -1)
        H_c = f._scalar_hessian("test", f._compute_mcstat_bb_compensator)
        np.testing.assert_allclose(np.asarray(H_c)[:2, :2], ref, rtol=1e-10)

    _, g, h = f.loss_val_grad_hess()
    assert np.abs(np.asarray(g)).max() < 1e-3
    assert np.linalg.eigvalsh(np.asarray(h)).min() > 0
    if models == "per-process":
        # nothing left to remove: the point is the ordinary fit's (up to M being
        # frozen at the reference while V moves), and only the covariance is
        # the sandwich A^-1 + A^-1 M A^-1
        f0 = fit_toy(fn, tmp_path, *args)
        sigma0 = sigma(np.linalg.inv(np.asarray(f0.loss_val_grad_hess()[2])))
        np.testing.assert_allclose(f.x.numpy(), f0.x.numpy(), atol=0.01 * sigma0)
        S = np.asarray(f.cov_mcstat_sandwich(h))
        Ainv = np.linalg.inv(np.asarray(h))
        np.testing.assert_allclose(S, Ainv + Ainv @ arr["M"] @ Ainv, rtol=1e-8)
