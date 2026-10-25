"""The batched likelihood: Fitter._compute_nll_batched at the B columns of
x [npar, B].

The likelihood has one implementation with the batch axis last; the existing
API (_compute_nll, the loss/grad/HVP wrappers, the shards) evaluates it at
self.x[:, None]. What these tests pin down:

- B columns give the same values as B evaluations at those points, for every
  layout the core covers (dense/sparse, symmetric/asymmetric, log-normal and
  normal systematics, masked channels, each bin-by-bin stat mode and type,
  chi2, frozen parameters, blinding, external terms);
- the gradient of their sum is the per-point gradient, column by column, which
  is what a batched minimizer differentiates;
- per-element observations (toys) give the fits to those observations;
- the terms that are not batched refuse B > 1 rather than disagree.
"""

import os
import tempfile
from types import SimpleNamespace

import hist
import numpy as np
import pytest
import scipy.sparse
import tensorflow as tf
from wums.sparse_hist import SparseHist

from rabbit import fitter, inputdata, tensorwriter
from rabbit import tfhelpers as tfh
from rabbit.param_models.helpers import load_model
from rabbit.param_models.param_model import Mu, compute_batched

from .test_sparse_fit import make_histograms, make_options

NBATCH = 4
RTOL = 1e-12


def _external_term(writer, sparse_hess):
    names = ["bkg_norm", "bkg_shape"]
    ax = hist.axis.StrCategory(names, name="params")
    grad = hist.Hist(ax, storage=hist.storage.Double())
    grad.values()[...] = [0.3, -0.2]
    hess_values = np.array([[2.0, 0.4], [0.4, 1.0]])
    # the writer reads a SparseHist's flat indices without flow bins
    ax0 = hist.axis.StrCategory(names, name="params0", overflow=False)
    ax1 = hist.axis.StrCategory(names, name="params1", overflow=False)
    if sparse_hess:
        hess = SparseHist(scipy.sparse.csr_array(hess_values), [ax0, ax1])
    else:
        hess = hist.Hist(ax0, ax1, storage=hist.storage.Double())
        hess.values()[...] = hess_values
    writer.add_external_likelihood_term(grad=grad, hess=hess, name="ext")


def _tensor(
    outdir,
    sparse=False,
    symmetric=True,
    systematic_type="log_normal",
    masked=False,
    external=None,
    data_covariance=False,
):
    h = make_histograms()
    writer = tensorwriter.TensorWriter(sparse=sparse, systematic_type=systematic_type)
    channels = ["ch0"] + (["masked"] if masked else [])
    for channel in channels:
        writer.add_channel(h["data"].axes, channel, masked=channel == "masked")
        if channel == "ch0":
            writer.add_data(h["data"], channel)
        writer.add_process(h["sig"], "sig", channel, signal=True)
        writer.add_process(h["bkg"], "bkg", channel)
        writer.add_norm_systematic("bkg_norm", "bkg", channel, 1.05)
        writer.add_norm_systematic("sig_norm", "sig", channel, 1.10)
        writer.add_systematic(
            [h["syst_up"], h["syst_dn"]],
            "bkg_shape",
            "bkg",
            channel,
            symmetrize="average" if symmetric else None,
        )
    if external is not None:
        _external_term(writer, sparse_hess=external == "sparse")
    if data_covariance:
        # correlated: D^1/2 (0.9 I + 0.1 J) D^1/2, positive definite
        root = np.sqrt(h["data"].values())
        corr = 0.9 * np.eye(len(root)) + 0.1
        writer.add_data_covariance(root[:, None] * corr * root[None, :])
    name = "batched"
    writer.write(outfolder=outdir, outfilename=name)
    return os.path.join(outdir, f"{name}.hdf5")


CASES = {
    "dense": {},
    "sparse": dict(tensor=dict(sparse=True)),
    "dense_asymmetric": dict(tensor=dict(symmetric=False)),
    "sparse_asymmetric": dict(tensor=dict(sparse=True, symmetric=False)),
    "dense_normal": dict(tensor=dict(systematic_type="normal")),
    "sparse_normal": dict(tensor=dict(sparse=True, systematic_type="normal")),
    "dense_masked": dict(tensor=dict(masked=True)),
    "sparse_masked": dict(tensor=dict(sparse=True, masked=True)),
    "no_bbb": dict(options=dict(noBinByBinStat=True)),
    "bbb_full": dict(options=dict(binByBinStatMode="full")),
    "bbb_full_gamma": dict(
        options=dict(binByBinStatMode="full", binByBinStatType="gamma")
    ),
    "bbb_dynamic": dict(options=dict(binByBinStatDynamic=True)),
    # per-process yields from the sparse tensor (densified)
    "sparse_bbb_full_gamma": dict(
        tensor=dict(sparse=True),
        options=dict(binByBinStatMode="full", binByBinStatType="gamma"),
    ),
    "bbb_normal_additive": dict(options=dict(binByBinStatType="normal-additive")),
    "bbb_normal_multiplicative": dict(
        options=dict(binByBinStatType="normal-multiplicative")
    ),
    "chisq": dict(options=dict(chisqFit=True)),
    "covariance": dict(
        tensor=dict(data_covariance=True), options=dict(covarianceFit=True)
    ),
    "frozen": dict(options=dict(freezeParameters=["bkg_norm"])),
    "blinded": dict(blinded=True),
    "external_dense": dict(tensor=dict(external="dense")),
    # SparseMatrixMatMul has no XLA kernel; "auto" only turns jit off for a
    # sparse tensor, not for a sparse external Hessian (the same on main)
    "external_sparse": dict(
        tensor=dict(external="sparse"), options=dict(jitCompile="off")
    ),
}


def _fitter(outdir, case):
    spec = CASES[case]
    indata = inputdata.FitInputData(_tensor(outdir, **spec.get("tensor", {})))
    model = load_model("Mu", indata, expectSignal=[("sig", 1.3)])
    f = fitter.Fitter(
        indata,
        model,
        make_options(**spec.get("options", {})),
        do_blinding=spec.get("blinded", False),
    )
    f.set_nobs(indata.data_obs)
    if spec.get("blinded"):
        f.set_blinding_offsets(True)
    return f


def _points(f, seed=1):
    """NBATCH points [npar, B] around the starting point; column 0 is x itself."""
    rng = np.random.default_rng(seed)
    x = f.x.numpy()
    cols = [x] + [x + 0.2 * rng.standard_normal(x.shape) for _ in range(NBATCH - 1)]
    return np.stack(cols, axis=-1)


def _single(f, X, fn):
    """fn() at each column of X, with self.x restored afterwards."""
    xsave = f.x.numpy()
    try:
        out = []
        for b in range(X.shape[-1]):
            f.x.assign(X[:, b])
            out.append(fn())
        return out
    finally:
        f.x.assign(xsave)


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("full_nll", [False, True])
def test_batch_equals_single_evaluations(case, full_nll):
    if full_nll and case in ("bbb_normal_multiplicative", "bbb_full"):
        # full mode picks normal-multiplicative for "automatic" on data
        pytest.skip("BinByBinStat.lbeta has no full_nll for normal-multiplicative")
    with tempfile.TemporaryDirectory() as tmp:
        f = _fitter(tmp, case)
        X = _points(f)
        batched = f._compute_nll_batched(tf.constant(X), full_nll=full_nll).numpy()
        single = _single(f, X, lambda: float(f._compute_nll(full_nll=full_nll).numpy()))
        assert batched.shape == (NBATCH,)
        np.testing.assert_allclose(batched, single, rtol=RTOL, atol=0)
        if not full_nll:
            # and the traced (jit-compiled where enabled) loss agrees too
            traced = _single(f, X, lambda: float(f.loss_val().numpy()))
            np.testing.assert_allclose(batched, traced, rtol=1e-11, atol=0)


@pytest.mark.parametrize(
    "case",
    ["dense", "sparse", "dense_asymmetric", "bbb_full_gamma", "blinded", "frozen"],
)
def test_gradient_of_the_sum_is_the_per_point_gradient(case):
    with tempfile.TemporaryDirectory() as tmp:
        f = _fitter(tmp, case)
        X = tf.constant(_points(f))
        with tf.GradientTape() as tape:
            tape.watch(X)
            total = tf.reduce_sum(f._compute_nll_batched(X))
        grads = tape.gradient(total, X).numpy()
        single = _single(f, X.numpy(), lambda: f.loss_val_grad()[1].numpy())
        for b in range(NBATCH):
            np.testing.assert_allclose(grads[:, b], single[b], rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("case", ["dense", "sparse", "bbb_full_gamma", "chisq"])
def test_per_element_observations_are_the_fits_to_them(case):
    """data with one column per element: B toys at B points in one call."""
    with tempfile.TemporaryDirectory() as tmp:
        f = _fitter(tmp, case)
        X = _points(f)
        nobs0 = f.nobs.numpy()
        rng = np.random.default_rng(7)
        nobs = np.stack(
            [np.maximum(rng.poisson(nobs0), 1.0) for _ in range(NBATCH)], axis=-1
        )
        lognobs = np.log(nobs)
        varnobs = nobs if f.chisqFit else None
        batched = f._compute_nll_batched(
            tf.constant(X), data=(tf.constant(nobs), varnobs, tf.constant(lognobs))
        ).numpy()

        single = []
        try:
            for b in range(NBATCH):
                f.set_nobs(tf.constant(nobs[:, b]))
                single.append(_single(f, X[:, b : b + 1], f._compute_nll)[0].numpy())
        finally:
            f.set_nobs(tf.constant(nobs0))
        np.testing.assert_allclose(batched, single, rtol=RTOL, atol=0)


def test_per_element_constraint_centres():
    """x0 with one column per element, as toys with randomized constraints."""
    with tempfile.TemporaryDirectory() as tmp:
        f = _fitter(tmp, "dense")
        X = _points(f)
        rng = np.random.default_rng(11)
        x00 = f.x0.numpy()
        x0 = x00[:, None] + 0.5 * rng.standard_normal((x00.shape[0], NBATCH))
        batched = f._compute_nll_batched(tf.constant(X), x0=tf.constant(x0)).numpy()
        single = []
        try:
            for b in range(NBATCH):
                f.x0.assign(x0[:, b])
                single.append(_single(f, X[:, b : b + 1], f._compute_nll)[0].numpy())
        finally:
            f.x0.assign(x00)
        np.testing.assert_allclose(batched, single, rtol=RTOL, atol=0)


@pytest.mark.parametrize("case", ["dense", "bbb_full_gamma"])
def test_unprofiled_batch_equals_single_evaluations(case):
    with tempfile.TemporaryDirectory() as tmp:
        f = _fitter(tmp, case)
        X = _points(f)
        batched = f._compute_nll_batched(tf.constant(X), profile=False).numpy()
        single = _single(f, X, lambda: float(f._compute_nll(profile=False).numpy()))
        np.testing.assert_allclose(batched, single, rtol=RTOL, atol=0)


class _BroadcastRateMu(Mu):
    """Mu whose rates broadcast over the processes: [nbins, 1] (one rate per
    bin, the shape of SaturatedProjectModel) or [1, 1]."""

    def __init__(self, indata, rate_shape, **kwargs):
        super().__init__(indata, **kwargs)
        self.rate_shape = rate_shape

    def compute(self, param, full=False):
        if self.rate_shape == "bins":
            nbins = self.indata.nbinsfull if full else self.indata.nbins
            ramp = tf.linspace(tf.constant(0.9, tf.float64), 1.1, nbins)
            return (1.0 + 0.1 * param[0]) * ramp[:, None]
        return tf.reshape(1.0 + param[0], [1, 1])


@pytest.mark.parametrize("rate_shape", ["bins", "scalar"])
@pytest.mark.parametrize("masked", [False, True])
def test_sparse_yields_broadcast_rates_like_dense(rate_shape, masked):
    """A rate without a process axis multiplies every process, sparse or dense."""
    with tempfile.TemporaryDirectory() as tmp:
        nlls = {}
        for sparse in (False, True):
            indata = inputdata.FitInputData(
                _tensor(os.path.join(tmp, str(sparse)), sparse=sparse, masked=masked)
            )
            model = _BroadcastRateMu(indata, rate_shape)
            f = fitter.Fitter(indata, model, make_options())
            f.set_nobs(indata.data_obs)
            X = _points(f)
            nlls[sparse] = f._compute_nll_batched(tf.constant(X)).numpy()
            single = _single(f, X, lambda: float(f._compute_nll().numpy()))
            np.testing.assert_allclose(nlls[sparse], single, rtol=RTOL, atol=0)
        np.testing.assert_allclose(nlls[True], nlls[False], rtol=1e-12, atol=0)


@pytest.mark.parametrize("symmetric", [True, False])
def test_mthetaalpha_keeps_the_column_count_without_systematics(symmetric):
    """With nsyst == 0 the batch width cannot be inferred from the empty axis."""
    stub = SimpleNamespace(indata=SimpleNamespace(nsyst=0, symmetric_tensor=symmetric))
    batched = fitter.Fitter._mthetaalpha(stub, tf.zeros([0, 3], tf.float64))
    single = fitter.Fitter._mthetaalpha(stub, tf.zeros([0], tf.float64))
    assert batched.shape == (0, 3)
    assert single.shape == (0, 1)


def test_mcstat_terms_refuse_the_batched_core(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        f = _fitter(tmp, "dense")
        monkeypatch.setattr(f, "_mcstat_terms_in_nll", lambda: True)
        with pytest.raises(NotImplementedError):
            f._compute_nll_batched(tf.constant(_points(f)))


def test_param_model_compute_batched_is_compute_per_column():
    with tempfile.TemporaryDirectory() as tmp:
        f = _fitter(tmp, "dense")
        model = f.param_model
        rng = np.random.default_rng(5)
        params = tf.constant(1.0 + 0.3 * rng.standard_normal((model.nparams, 3)))
        batched = model.compute_batched(params).numpy()
        for b in range(3):
            np.testing.assert_array_equal(
                batched[..., b], model.compute(params[:, b]).numpy()
            )
        # B = 1 is one direct call of compute()
        np.testing.assert_array_equal(
            model.compute_batched(params[:, :1]).numpy()[..., 0],
            model.compute(params[:, 0]).numpy(),
        )


def test_compute_batched_of_a_model_with_compute_only():
    """Duck-typed param models (no ParamModel base) only define compute()."""

    class ComputeOnly:
        def compute(self, param, full=False):
            return 2.0 * param[None, :]

    model = ComputeOnly()
    params = tf.reshape(tf.range(6.0, dtype=tf.float64), [2, 3])
    batched = compute_batched(model, params).numpy()
    for b in range(3):
        np.testing.assert_array_equal(batched[..., b], model.compute(params[:, b]))


def test_map_batch_shares_size_one_arguments_and_keeps_none():
    a = tf.reshape(tf.range(6.0), [2, 3])  # batch of 3
    shared = tf.constant([[10.0], [20.0]])  # [2, 1], one value for all elements

    def fn(a, s, missing):
        assert missing is None
        return a + s, None, tf.reduce_sum(a)

    plus, none, total = tfh.map_batch(fn, (a, shared, None))
    assert none is None
    np.testing.assert_array_equal(plus.numpy(), (a + shared).numpy())
    np.testing.assert_array_equal(total.numpy(), tf.reduce_sum(a, axis=0).numpy())

    plus1, _, total1 = tfh.map_batch(fn, (a[:, :1], shared, None))
    np.testing.assert_array_equal(plus1.numpy(), (a[:, :1] + shared).numpy())
    assert total1.shape == (1,)
