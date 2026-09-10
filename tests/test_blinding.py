"""
Test that blinding is a change of variables that leaves the PHYSICS alone.

rabbit blinds by reparametrising the likelihood, not by masking output: the
getters apply the offset on the way INTO ParamModel.compute() and the constraint
term, so the model and the NLL see the physical value while ``fitter.x`` -- the
minimizer's coordinate, and what gets written out -- is the blinded one.

The two forms are not interchangeable, which is what these tests pin down:

* MULTIPLICATIVE (the default for POIs) suits a signal strength centred at 1
  that scales yields. But the reported coordinate is then ``poi_true / offset``,
  so the curvature scales as ``offset**2`` and the reported uncertainty, the POI
  row of the covariance and every impact on that POI are divided by the random
  factor. Only the RELATIVE uncertainty survives.
* ADDITIVE, which a model opts into with ``blind_additive = True``, is a
  translation. Its Jacobian is the identity, so the covariance, the
  uncertainties and the impacts come out EXACTLY unblinded while the central
  value is still hidden.

Every check below is an INVARIANCE, so no test prints, returns or asserts on an
offset value.
"""

import os
import tempfile
from types import SimpleNamespace

import hist
import numpy as np
import tensorflow as tf

from rabbit import fitter, inputdata, tensorwriter
from rabbit.param_models.param_model import ParamModel

# Deliberately NON-ZERO. A zero default would satisfy the start-invariance test
# by accident -- ``0 * offset == 0`` for the multiplicative form -- which is
# exactly the accident this machinery replaces with a guarantee.
START = 0.3


class ToyModel(ParamModel):
    """One POI scaling the signal, linear so the fit solves exactly."""

    def __init__(self, indata, blind_additive=False):
        super().__init__(indata)
        self.npoi = 1
        self.npou = 0
        self.params = np.array([b"alphaS"])
        self.xparamdefault = tf.constant([START], dtype=indata.dtype)
        self.is_linear = True
        self.allowNegativeParam = True
        if blind_additive:
            self.blind_additive = True

    def compute(self, param, full=False):
        # No numpy on `param`: compute() runs inside a tf.function, where it is
        # symbolic. The tests assert observable consequences instead.
        nproc = self.indata.nproc
        col = tf.reshape(1.0 + 0.1 * param[0], [1, 1])
        return tf.concat([col, tf.ones([1, nproc - 1], dtype=col.dtype)], axis=1)


def make_tensor(path):
    np.random.seed(1234)
    ax = hist.axis.Regular(20, -5, 5, name="x")
    h_data = hist.Hist(ax, storage=hist.storage.Double())
    h_sig = hist.Hist(ax, storage=hist.storage.Weight())
    h_bkg = hist.Hist(ax, storage=hist.storage.Weight())
    h_data.fill(
        np.concatenate([np.random.normal(0, 1, 8000), np.random.uniform(-5, 5, 4000)])
    )
    h_sig.fill(np.random.normal(0, 1, 8000))
    h_bkg.fill(np.random.uniform(-5, 5, 4000))

    w = tensorwriter.TensorWriter()
    w.add_channel(h_data.axes, "ch0")
    w.add_data(h_data, "ch0")
    w.add_process(h_sig, "sig", "ch0", signal=True)
    w.add_process(h_bkg, "bkg", "ch0", signal=False)
    # One ordinary constrained systematic, so the theta block is not empty.
    w.add_norm_systematic("bkgNorm", ["bkg"], "ch0", 1.05)
    w.write(outfolder=os.path.dirname(path), outfilename=os.path.basename(path))


def make_options(**kwargs):
    defaults = dict(
        earlyStopping=-1,
        noBinByBinStat=True,
        binByBinStatMode="lite",
        binByBinStatType="automatic",
        covarianceFit=False,
        chisqFit=False,
        diagnostics=False,
        minimizerMethod="trust-krylov",
        prefitUnconstrainedNuisanceUncertainty=0.0,
        freezeParameters=[],
        setConstraintMinimum=[],
        unblind=[],
        blindingGroup=[],
        maxRestarts=-1,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def build(path, blind_additive, do_blinding, **opts):
    ind = inputdata.FitInputData(path)
    model = ToyModel(ind, blind_additive=blind_additive)
    f = fitter.Fitter(ind, model, make_options(**opts), do_blinding=do_blinding)
    return ind, model, f


def _asimov(f):
    """Asimov data at the current point.

    Required for the Hessian checks to mean anything: with nobs = 0 the Poisson
    term ``sum(nexp - nobs*log nexp)`` collapses to ``sum(nexp)``, which is
    LINEAR in the yields, so the POI curvature is exactly zero and a comparison
    of Hessians passes vacuously via ``isclose(0, 0)``.
    """
    return f.expected_yield()


def test_model_sees_physical_while_x_is_blinded(path):
    """compute() gets the physical value; fitter.x holds the blinded one."""
    _, _, fu = build(path, True, False)
    fu.defaultassign()
    y_unblinded = fu.expected_yield().numpy()

    _, model, fb = build(path, True, True)
    fb.defaultassign()
    fb.set_blinding_offsets(True)
    # Zero the THETA offsets to isolate the POI: theta starts at 0, so arming
    # puts the physical NOIs at their offsets and moves the yields for reasons
    # that have nothing to do with the POI under test.
    fb._blinding_offsets_theta.assign(np.zeros(fb.indata.nsyst, dtype=np.float64))
    y_blinded = fb.expected_yield().numpy()

    assert np.isclose(float(fb.get_poi()[0].numpy()), START, rtol=0, atol=1e-12)
    assert np.allclose(y_blinded, y_unblinded, rtol=1e-12, atol=0)
    assert not np.isclose(float(fb.x[0].numpy()), START, rtol=0, atol=1e-9)


def test_additive_leaves_hessian_exactly_unblinded(path):
    """sigma = sqrt(diag(H^-1)), so 'sigma unblinded' IS 'H unchanged'."""

    def loss_and_hess(blind_additive, do_blinding, asimov):
        _, _, f = build(path, blind_additive, do_blinding)
        f.defaultassign()
        if do_blinding:
            f.set_blinding_offsets(True)
            f._blinding_offsets_theta.assign(np.zeros(f.indata.nsyst, dtype=np.float64))
        f.set_nobs(asimov)
        loss, _, hess = f.loss_val_grad_hess()
        return float(loss.numpy()), hess.numpy()

    _, _, f_ref = build(path, True, False)
    f_ref.defaultassign()
    asimov = _asimov(f_ref)

    l_u, h_u = loss_and_hess(True, False, asimov)
    l_a, h_a = loss_and_hess(True, True, asimov)

    # Guard against a vacuous pass before comparing.
    assert abs(h_u[0, 0]) > 1e-6, f"no POI curvature to compare: {h_u[0, 0]}"
    assert np.isclose(l_a, l_u, rtol=0, atol=1e-9)
    assert np.isclose(h_a[0, 0], h_u[0, 0], rtol=1e-10)
    assert np.allclose(h_a, h_u, rtol=1e-10, atol=0)

    # The defect being fixed: the multiplicative path scales the same element.
    _, h_m = loss_and_hess(False, True, asimov)
    assert not np.isclose(h_m[0, 0], h_u[0, 0], rtol=1e-6)


def test_physical_start_invariant_under_arming(path):
    """Arming must not move the physical point, and must be idempotent."""
    _, _, f = build(path, True, True)
    f.defaultassign()
    assert np.isclose(float(f.get_poi()[0].numpy()), START, rtol=0, atol=1e-14)

    f.set_blinding_offsets(True)
    p_armed = float(f.get_poi()[0].numpy())
    assert np.isclose(p_armed, START, rtol=0, atol=1e-12)
    assert not np.isclose(float(f.x[0].numpy()), p_armed, rtol=0, atol=1e-9)

    f.set_blinding_offsets(True)  # idempotent: shifts by zero
    assert np.isclose(float(f.get_poi()[0].numpy()), p_armed, rtol=0, atol=1e-14)

    f.set_blinding_offsets(False)
    assert np.isclose(float(f.get_poi()[0].numpy()), START, rtol=0, atol=1e-12)


def test_x0_untouched_by_arming(path):
    """x0 is the model frame and must NOT be shifted; cheapest guard against
    someone 'symmetrising' the compensation later."""
    _, _, f = build(path, True, True)
    f.defaultassign()
    before = f.x0.numpy().copy()
    f.set_blinding_offsets(True)
    assert np.array_equal(before, f.x0.numpy())


def test_multiplicative_path_unchanged(path):
    """A model that does not opt in keeps exactly its current arithmetic."""
    _, _, f = build(path, False, True)
    f.defaultassign()
    f.set_blinding_offsets(True)
    assert float(f._blinding_offsets_poi_add[0].numpy()) == 0.0
    assert np.isclose(
        float(f.get_poi()[0].numpy()),
        float(f.x[0].numpy()) * float(f._blinding_offsets_poi[0].numpy()),
        rtol=1e-14,
    )
    # x is NOT frame-shifted for a multiplicative model.
    assert np.isclose(float(f.x[0].numpy()), START, rtol=0, atol=1e-14)


def test_determinism_across_fitters(path):
    """Same input, independent fitters: identical offsets, without printing one."""
    _, _, f1 = build(path, True, True)
    _, _, f2 = build(path, True, True)
    for f in (f1, f2):
        f.defaultassign()
        f.set_blinding_offsets(True)
    d1 = f1.get_poi().numpy() - f1.x[:1].numpy()
    d2 = f2.get_poi().numpy() - f2.x[:1].numpy()
    np.testing.assert_array_equal(d1, d2)
    assert not np.allclose(d1, 0.0)  # non-trivial


def test_unblind_disables_the_offset(path):
    """--unblind on the POI leaves the coordinate and the physical value equal."""
    _, _, f = build(path, True, True, unblind=["alphaS"])
    f.defaultassign()
    f.set_blinding_offsets(True)
    assert np.isclose(
        float(f.get_poi()[0].numpy()), float(f.x[0].numpy()), rtol=0, atol=1e-14
    )


def test_additive_requires_allow_negative_param(path):
    """With the squared storage an additive offset could hand compute() a
    negative POI, so the combination must be refused."""
    ind = inputdata.FitInputData(path)
    model = ToyModel(ind, blind_additive=True)
    model.allowNegativeParam = False
    try:
        fitter.Fitter(ind, model, make_options(), do_blinding=True)
    except ValueError as exc:
        assert "allowNegativeParam" in str(exc)
    else:
        raise AssertionError("expected a refusal for blind_additive + squared storage")


def main():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "blinding_tensor.hdf5")
        make_tensor(path)
        tests = [
            test_model_sees_physical_while_x_is_blinded,
            test_additive_leaves_hessian_exactly_unblinded,
            test_physical_start_invariant_under_arming,
            test_x0_untouched_by_arming,
            test_multiplicative_path_unchanged,
            test_determinism_across_fitters,
            test_unblind_disables_the_offset,
            test_additive_requires_allow_negative_param,
        ]
        failed = []
        for t in tests:
            try:
                t(path)
                print(f"  OK   {t.__name__}")
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL {t.__name__}: {type(exc).__name__}: {exc}")
                failed.append(t.__name__)
        if failed:
            print(f"SOME CHECKS FAILED: {failed}")
            raise SystemExit(1)
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
