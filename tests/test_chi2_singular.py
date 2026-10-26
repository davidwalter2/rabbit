"""The postfit chi2 of a mapping whose residual covariance is exactly singular.

Bins that obey an exact linear relation (two projections of one channel, ABCD
regions) give a singular residual covariance. The solve then fails or not
depending on rounding; expected_events falls back to the pseudo inverse.
Duplicating a projection makes the covariance singular for any rounding.
"""

import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from rabbit import fitter, inputdata, tensorwriter
from rabbit.mappings import helpers as mh
from rabbit.mappings import mapping as mp
from rabbit.param_models.helpers import load_model
from rabbit.tfhelpers import edmval_cov

from .test_sparse_fit import make_histograms, make_options


def _fitter(outdir):
    """A one-channel dense fit: signal, background with a norm and a shape
    systematic, bin-by-bin stat."""
    h = make_histograms()
    writer = tensorwriter.TensorWriter()
    writer.add_channel(h["data"].axes, "ch0")
    writer.add_data(h["data"], "ch0")
    writer.add_process(h["sig"], "sig", "ch0", signal=True)
    writer.add_process(h["bkg"], "bkg", "ch0")
    writer.add_norm_systematic("bkg_norm", "bkg", "ch0", 1.05)
    writer.add_norm_systematic("sig_norm", "sig", "ch0", 1.10)
    writer.add_systematic(
        [h["syst_up"], h["syst_dn"]], "bkg_shape", "bkg", "ch0", symmetrize="average"
    )
    writer.write(outfolder=outdir, outfilename="chi2_singular")
    indata = inputdata.FitInputData(os.path.join(outdir, "chi2_singular.hdf5"))
    model = load_model("Mu", indata, expectSignal=[("sig", 1.3)])
    f = fitter.Fitter(indata, model, make_options())
    f.set_nobs(indata.data_obs)
    return f


@pytest.fixture(scope="module")
def postfit():
    with tempfile.TemporaryDirectory() as outdir:
        f = _fitter(outdir)
        # the test data are exactly sig / 0.9 + bkg, which the fit reproduces: fluctuate
        nobs = np.random.default_rng(3).poisson(f.indata.data_obs.numpy())
        f.set_nobs(tf.constant(nobs, dtype=f.indata.dtype))
        f.minimize()
        _, grad, hess = f.loss_val_grad_hess()
        _, cov = edmval_cov(grad, hess)
        f.cov.assign(cov)
        yield f


def test_singular_covariance_uses_the_pseudo_inverse(postfit):
    f = postfit
    single = mh.load_mapping("Project", f.indata, "ch0", "x")
    twice = mp.CompositeMapping(
        [single, mh.load_mapping("Project", f.indata, "ch0", "x")]
    )

    # the plain solve cannot invert the duplicated covariance
    with pytest.raises(tf.errors.InvalidArgumentError):
        f.chi2(twice.compute_flat, twice._get_data, twice.ndf_reduction, profile=True)

    _, (*_, chi2_single, ndf_single) = f.expected_events(single, compute_chi2=True)
    _, (*_, chi2_twice, ndf_twice) = f.expected_events(twice, compute_chi2=True)

    # [r, r]^T pinv([[C, C], [C, C]]) [r, r] = r^T C^-1 r
    np.testing.assert_allclose(chi2_twice, chi2_single, rtol=1e-8)
    assert int(ndf_single) == 20
    assert int(ndf_twice) == 40  # the ndf is not changed by the fallback


def test_pseudo_inverse_fallback_still_fails_on_a_non_finite_covariance(postfit):
    """The fallback covers an exactly singular covariance, not a NaN one."""
    res = tf.ones([2], dtype=postfit.indata.dtype)
    singular = tf.ones([2, 2], dtype=postfit.indata.dtype)
    chi2, ndf = postfit._chi2(res, singular, pseudo_inverse=True)
    np.testing.assert_allclose(chi2, 1.0)  # [1, 1] pinv([[1, 1], [1, 1]]) [1, 1]
    assert int(ndf) == 2
    nan = tf.constant([[1.0, float("nan")], [float("nan"), 1.0]], dtype=res.dtype)
    with pytest.raises(tf.errors.InvalidArgumentError, match="not finite"):
        postfit._chi2(res, nan, pseudo_inverse=True)
