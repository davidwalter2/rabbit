"""Bin-by-bin stat toys draw with the width of the constraint lbeta uses.

lbeta is kstat * (beta - beta0 * log(beta)) for gamma, 0.5 * kstat *
(beta - beta0)^2 for normal-multiplicative and 0.5 * (beta - beta0)^2 for
normal-additive (beta in units of sqrt(varbeta)). Frequentist toys draw the
centre beta0 around beta, Bayesian toys beta around beta0, from those
constraints. For normal-multiplicative the width is 1/sqrt(kstat) =
sqrt(varbeta) / sumw; the toys drew with sqrt(varbeta), the absolute MC stat
uncertainty of the yield, a factor sumw too wide (sqrt(varbeta) was the
relative width while varbeta was 1/kstat). Masked entries have no constraint
and keep their centre.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from rabbit.bbstat.bbstat import BinByBinStat

# Four bins, three processes: "sig" (weight 5, kstat 4 to 400), "bkg"
# (weight 0.5, kstat 400) and "flat" without MC stat uncertainty (masked in
# full mode).
NSIG = np.array([4.0, 25.0, 100.0, 400.0])
SUMW = np.stack([5.0 * NSIG, np.full(4, 200.0), np.full(4, 40.0)], axis=-1)
SUMW2 = np.stack([25.0 * NSIG, np.full(4, 100.0), np.zeros(4)], axis=-1)
TYPES = ["gamma", "normal-multiplicative", "normal-additive"]
NDRAW = 2000


def _bbstat(stat_type, mode):
    indata = SimpleNamespace(
        sumw=tf.constant(SUMW),
        sumw2=tf.constant(SUMW2),
        nbins=SUMW.shape[0],
        nbinsmasked=0,
        nbinsfull=SUMW.shape[0],
        dtype=tf.float64,
        norm=None,
        betavar=None,
    )
    options = SimpleNamespace(
        noBinByBinStat=False,
        binByBinStatMode=mode,
        binByBinStatType=stat_type,
        minBBKstat=0.0,
    )
    return BinByBinStat(
        indata, options, chisqFit=False, covarianceFit=False, data_cov_inv=None
    )


def _draws(bb, randomize, variable):
    """NDRAW draws of `variable` by `randomize`, from the default state."""
    tf.random.set_seed(1)
    out = []
    for _ in range(NDRAW):
        bb.beta0_default_assign()
        bb.beta_default_assign()
        randomize()
        out.append(variable.numpy())
    return np.array(out)


def _check_spread(bb, stat_type, draws, centre, width):
    """Sample mean and standard deviation per entry against the centre and
    width. Masked entries have no constraint: normal-multiplicative leaves
    them at their centre, gamma draws them with its placeholder kstat (not
    checked here), normal-additive does not use betamask."""
    mask = bb.betamask.numpy()
    if stat_type == "normal-multiplicative":
        assert (draws[:, mask] == centre[mask][None]).all()
    if stat_type == "normal-additive":
        mask = np.zeros_like(mask)
    mean, std = draws.mean(axis=0)[~mask], draws.std(axis=0)[~mask]
    width, centre = width[~mask], centre[~mask]
    np.testing.assert_allclose(std, width, rtol=0.06)
    assert np.all(np.abs(mean - centre) < 5 * width / np.sqrt(NDRAW))


@pytest.mark.parametrize("mode", ["full", "lite"])
@pytest.mark.parametrize("stat_type", TYPES)
def test_frequentist_centres(stat_type, mode):
    bb = _bbstat(stat_type, mode)
    kstat = bb.kstat.numpy()
    centre = bb.default_beta0().numpy()
    # beta0 around beta (the default): Poisson(kstat * beta) / kstat for gamma
    width = {
        "gamma": np.sqrt(centre / kstat),
        "normal-multiplicative": 1.0 / np.sqrt(kstat),
        "normal-additive": np.ones_like(kstat),
    }[stat_type]
    draws = _draws(bb, bb.randomize_frequentist, bb.beta0)
    _check_spread(bb, stat_type, draws, centre, width)


@pytest.mark.parametrize("mode", ["full", "lite"])
@pytest.mark.parametrize("stat_type", TYPES)
def test_bayesian_values(stat_type, mode):
    bb = _bbstat(stat_type, mode)
    kstat = bb.kstat.numpy()
    beta0 = bb.default_beta0().numpy()
    # beta around beta0: Gamma(kstat * beta0 + 1) / kstat for gamma
    centre, width = {
        "gamma": ((kstat * beta0 + 1) / kstat, np.sqrt(kstat * beta0 + 1) / kstat),
        "normal-multiplicative": (beta0, 1.0 / np.sqrt(kstat)),
        "normal-additive": (beta0, np.ones_like(kstat)),
    }[stat_type]
    draws = _draws(bb, bb.randomize_bayes, bb.beta)
    _check_spread(bb, stat_type, draws, centre, width)
