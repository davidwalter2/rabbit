"""Response basis of the double differential SME model (sme_double_differential.py).

For each (mll, rapidity) bin, the SM cross section (summed over SM_QUARKS) and its
derivatives w.r.t. the lab frame coefficients (c00, c33, c03) for c (cL = cR) and
d (cL = -cR) type coefficients, per quark of LIV_QUARKS. Kept free of tensorflow so it can be computed in parallel
(e.g. by scripts/plotting/liv_pheno_fit_rapidity.py) before running the fit.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from rabbit.param_models.liv_rapidity.functions import S, dsig_dyll, integrate_xsec_yll


def bin_xsec(qid, int_bin, cL, cR):
    """cross section of one (mll, rapidity) bin (Qlo, Qhi, ylo, yhi). A point bin, Qlo == Qhi,
    is dsigma/dm_ll evaluated exactly at m_ll = Qlo (= 2 Q dsigma/dQ^2), integrated over
    ylo < y < yhi"""
    Qlo, Qhi, ylo, yhi = int_bin
    if Qlo == Qhi:
        return 2.0 * Qlo * dsig_dyll(qid, Qlo, ylo, yhi, cL, cR)
    return integrate_xsec_yll(qid, Qlo, Qhi, ylo, yhi, cL, cR)

# quarks with SME responses (d, u, s, c, b)
LIV_QUARKS = [1, 2, 3, 4, 5]
# quarks summed in the SM cross section, the denominator of the relative SME shift
# (d, u, s, c, b, t; the 5 flavour PDF set has no top, which therefore adds zero)
SM_QUARKS = [1, 2, 3, 4, 5, 6]
EPS = 1e-6
PDF_NAME = "NNPDF31_nnlo_as_0118"

# full dilepton rapidity range at the lowest mll (15 GeV) of the fits, |y| < ln(sqrt(S) / Q),
# used for channels without a rapidity axis that are integrated over all rapidities
YLL_FULL = float(0.5 * np.log(S / 15.0**2))

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "precomputed_sigma"
)


def _fmt(x):
    """number for a file name: 91.1879 -> 91p1879, -2.5 -> m2p5"""
    return f"{x:g}".replace("-", "m").replace(".", "p")


def _range_tag(edges, name):
    """'<n><name><lo>to<hi>' for n bins (lo, hi); '<name><x>' for a single point bin (x, x)"""
    if len(edges) == 1 and edges[0][0] == edges[0][1]:
        return f"{name}{_fmt(edges[0][0])}"
    lo, hi = min(e[0] for e in edges), max(e[1] for e in edges)
    return f"{len(edges)}{name}{_fmt(lo)}to{_fmt(hi)}"


def bin_tag(int_bin):
    """human readable name of one (Qlo, Qhi, ylo, yhi) bin, e.g. mll66to71_yllm2p5to0"""
    Qlo, Qhi, ylo, yhi = int_bin
    mll = f"mll{_fmt(Qlo)}" if Qlo == Qhi else f"mll{_fmt(Qlo)}to{_fmt(Qhi)}"
    yll = "yllall" if (ylo, yhi) == (-YLL_FULL, YLL_FULL) else f"yll{_fmt(ylo)}to{_fmt(yhi)}"
    return f"{mll}_{yll}"


def basis_tag(int_bins):
    """human readable name of a list of (Qlo, Qhi, ylo, yhi) bins: number and range of the
    mll and rapidity bins, e.g. 10mll66to116_2yllm2p5to2p5, mll91p1879_yllall"""
    mll = sorted({(b[0], b[1]) for b in int_bins})
    yll = sorted({(b[2], b[3]) for b in int_bins})
    ytag = "yllall" if yll == [(-YLL_FULL, YLL_FULL)] else _range_tag(yll, "yll")
    return f"{_range_tag(mll, 'mll')}_{ytag}"


def basis_cache_file(int_bins):
    """int_bins: list of (Qlo, Qhi, ylo, yhi) python floats, in model order. The name gives
    the number and range of the mll and rapidity bins and the PDF; the bins themselves are
    stored in the file and checked on loading (bins with the same name but other edges)."""
    return os.path.join(CACHE_DIR, f"liv_basis_{basis_tag(int_bins)}_{PDF_NAME}.npz")


def sm_xsec(int_bin, quarks):
    """SM cross section of one (mll, rapidity) bin summed over the quarks"""
    Qlo, Qhi, ylo, yhi = int_bin
    ykin = 0.5 * np.log(S / Qlo**2)
    ylo, yhi = max(ylo, -ykin), min(yhi, ykin)
    if ylo >= yhi:
        return 0.0
    return sum(bin_xsec([q], (Qlo, Qhi, ylo, yhi), [0, 0, 0], [0, 0, 0]) for q in quarks)


def quark_response(int_bin, quarks):
    """SM cross section summed over the quarks and response [mode (c/d), direction
    (c00/c33/c03), quark] of one bin"""
    Qlo, Qhi, ylo, yhi = int_bin
    resp = np.zeros((2, 3, len(quarks)))
    # the cross section vanishes beyond the kinematic limit |y| < ln(sqrt(S) / Q)
    ykin = 0.5 * np.log(S / Qlo**2)
    ylo, yhi = max(ylo, -ykin), min(yhi, ykin)
    if ylo >= yhi:
        return 0.0, resp
    sm = 0.0
    b = (Qlo, Qhi, ylo, yhi)
    for iq, q in enumerate(quarks):
        s0 = bin_xsec([q], b, [0, 0, 0], [0, 0, 0])
        sm += s0
        for k in range(3):
            e = [0.0, 0.0, 0.0]
            e[k] = EPS
            en = [-v for v in e]
            resp[0, k, iq] = (bin_xsec([q], b, e, e) - s0) / EPS
            resp[1, k, iq] = (bin_xsec([q], b, e, en) - s0) / EPS
    return sm, resp


def _sm_extra(int_bin):
    return sm_xsec(int_bin, [q for q in SM_QUARKS if q not in LIV_QUARKS])


def bin_response(int_bin):
    """sm cross section and response [mode (c/d), direction (c00/c33/c03), quark] of one bin"""
    sm, resp = quark_response(int_bin, LIV_QUARKS)
    # the cross section is a sum over all quarks, also those without SME responses
    return sm + _sm_extra(int_bin), resp


def compute_basis(int_bins, jobs=1, logger=None):
    """returns sm_total [nbins], resp [mode, direction, quark, nbins]"""
    sm_total = np.zeros(len(int_bins))
    resp = np.zeros((2, 3, len(LIV_QUARKS), len(int_bins)))

    def _fill(results):
        for ib, (sm, r) in enumerate(results):
            sm_total[ib], resp[..., ib] = sm, r
            if logger is not None and ((ib + 1) % 100 == 0 or ib + 1 == len(int_bins)):
                logger.info(f"LIV: bin {ib + 1}/{len(int_bins)} done")

    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            _fill(ex.map(bin_response, int_bins, chunksize=4))
    else:
        _fill(map(bin_response, int_bins))
    return sm_total, resp


def load_or_compute_basis(int_bins, jobs=1, logger=None):
    cache_file = basis_cache_file(int_bins)
    if os.path.exists(cache_file):
        cached = np.load(cache_file)
        if "int_bins" in cached.files and np.array_equal(
            cached["int_bins"], np.array(int_bins, dtype=float)
        ):
            return cached["sm_total"], cached["resp"]
        if logger is not None:
            logger.warning(
                f"LIV: {cache_file} was computed for other bins with the same name, recomputing"
            )
    if logger is not None:
        logger.info(
            f"LIV: integrating response basis for {len(int_bins)} (mll, rapidity) bins, will cache to {cache_file}"
        )
    sm_total, resp = compute_basis(int_bins, jobs, logger)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(
        cache_file, sm_total=sm_total, resp=resp, int_bins=np.array(int_bins, dtype=float)
    )
    return sm_total, resp
