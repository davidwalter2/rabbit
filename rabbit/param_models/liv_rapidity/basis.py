"""Response basis of the double differential SME model (sme_double_differential.py).

For each (mll, rapidity) bin, the SM cross section (summed over SM_QUARKS) and its
derivatives w.r.t. the lab frame coefficients (c00, c33, c03) for c (cL = cR) and
d (cL = -cR) type coefficients, per quark of LIV_QUARKS. Kept free of tensorflow so it can be computed in parallel
(e.g. by scripts/plotting/liv_pheno_fit_rapidity.py) before running the fit.
"""

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

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
# quarks with SME responses before bottom was added: their responses in older caches are
# reused, so that only the bottom responses are integrated
PREV_LIV_QUARKS = [1, 2, 3, 4]
# all earlier quark sets whose caches can be extended, newest first (before charm: d, u, s)
PREV_LIV_QUARK_SETS = [PREV_LIV_QUARKS, [1, 2, 3]]
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


def basis_cache_file(int_bins, sm_quarks=SM_QUARKS, liv_quarks=LIV_QUARKS):
    """int_bins: list of (Qlo, Qhi, ylo, yhi) python floats, in model order"""
    key = (list(int_bins), list(liv_quarks), EPS, PDF_NAME)
    if list(sm_quarks) != list(liv_quarks):
        key = key + (list(sm_quarks),)
    cache_key = hashlib.md5(repr(key).encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"liv_basis_{cache_key}.npz")


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


def _extend_bin(int_bin, new_quarks, sm_quarks):
    """SM cross section of sm_quarks and response of new_quarks of one bin"""
    _, resp = quark_response(int_bin, new_quarks)
    return sm_xsec(int_bin, sm_quarks), resp


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


def _extend_cached_basis(int_bins, jobs, logger):
    """basis from a cache of an earlier quark set (PREV_LIV_QUARK_SETS; SM cross section
    summed over SM_QUARKS, or over the earlier quarks only for the oldest caches),
    integrating only the responses of the added quarks. None if there is no such cache."""
    for prev_quarks in PREV_LIV_QUARK_SETS:
        basis = _extend_from(int_bins, prev_quarks, jobs, logger)
        if basis is not None:
            return basis
    return None


def _extend_from(int_bins, prev_quarks, jobs, logger):
    new_quarks = [q for q in LIV_QUARKS if q not in prev_quarks]
    for old_sm_quarks in (SM_QUARKS, prev_quarks):
        old_file = basis_cache_file(int_bins, sm_quarks=old_sm_quarks, liv_quarks=prev_quarks)
        if not os.path.exists(old_file):
            continue
        extra_sm = [q for q in SM_QUARKS if q not in old_sm_quarks]
        if logger is not None:
            logger.info(
                f"LIV: reusing the responses of quarks {prev_quarks} from {old_file}, "
                f"integrating quarks {new_quarks}"
                + (f" and adding quarks {extra_sm} to the SM cross section" if extra_sm else "")
            )
        func = partial(_extend_bin, new_quarks=new_quarks, sm_quarks=extra_sm)
        if jobs > 1:
            with ProcessPoolExecutor(jobs) as ex:
                results = list(ex.map(func, int_bins, chunksize=4))
        else:
            results = [func(b) for b in int_bins]
        cached = np.load(old_file)
        sm_total = cached["sm_total"] + np.array([r[0] for r in results])
        resp = np.zeros((2, 3, len(LIV_QUARKS), len(int_bins)))
        for iq, q in enumerate(prev_quarks):
            resp[:, :, LIV_QUARKS.index(q)] = cached["resp"][:, :, iq]
        new_resp = np.stack([r[1] for r in results], axis=-1)  # [mode, direction, quark, nbins]
        for iq, q in enumerate(new_quarks):
            resp[:, :, LIV_QUARKS.index(q)] = new_resp[:, :, iq]
        return sm_total, resp
    return None


def load_or_compute_basis(int_bins, jobs=1, logger=None):
    cache_file = basis_cache_file(int_bins)
    if os.path.exists(cache_file):
        cached = np.load(cache_file)
        return cached["sm_total"], cached["resp"]
    basis = _extend_cached_basis(int_bins, jobs, logger)
    if basis is None:
        if logger is not None:
            logger.info(
                f"LIV: integrating response basis for {len(int_bins)} (mll, rapidity) bins, will cache to {cache_file}"
            )
        basis = compute_basis(int_bins, jobs, logger)
    sm_total, resp = basis
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache_file, sm_total=sm_total, resp=resp)
    return sm_total, resp
