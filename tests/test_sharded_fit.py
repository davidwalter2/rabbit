"""Tests for multi-device (bins-sharded) likelihood evaluation.

Without GPUs the shards all land on the CPU (sharding.select_devices
falls back with a warning), which exercises every piece of the sharded
machinery -- the bin slicing, the per-shard BinByBinStat, the borrowed
methods, the per-shard tapes, the partial-sum combiners -- with no
device separation. Device *placement* correctness was established
separately on a 4-GPU node (see rabbit/sharding.py).

The bar everywhere is near-bitwise agreement with the single-device
path: sharding is a memory layout, not an approximation.
"""

import tempfile

import numpy as np
import pytest
import tensorflow as tf

from rabbit import fitter, inputdata
from rabbit.param_models.helpers import load_model

from .test_sparse_fit import make_options, make_test_tensor


def _make_fitter(filename, ndevices=1, **kw):
    indata_obj = inputdata.FitInputData(filename, host_memory=ndevices > 1)
    param_model = load_model("Mu", indata_obj)
    options = make_options(nDevices=ndevices, **kw)
    # pass the kwargs rabbit_fit passes, so the factory can never silently
    # drop one again (it did once: globalImpactsFromJVP)
    f = fitter.make_fitter(
        indata_obj, param_model, options, do_blinding=False, globalImpactsFromJVP=True
    )
    f.set_nobs(indata_obj.data_obs)
    return f


RTOL = 1e-12


@pytest.mark.parametrize("ndevices", [2, 3])
def test_sharded_loss_grad_hvp_hess_match(ndevices):
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = make_test_tensor(tmpdir)
        f1 = _make_fitter(fname, 1)
        fn = _make_fitter(fname, ndevices)
        assert len(fn.shards) == ndevices

        # move off the trivial starting point so gradients are non-trivial
        rng = np.random.default_rng(3)
        xval = f1.x.numpy() + 0.1 * rng.standard_normal(f1.x.shape[0])
        f1.x.assign(xval)
        fn.x.assign(xval)

        v1, g1 = f1.loss_val_grad()
        vn, gn = fn.loss_val_grad()
        np.testing.assert_allclose(float(vn), float(v1), rtol=RTOL)
        np.testing.assert_allclose(gn.numpy(), g1.numpy(), rtol=1e-10, atol=1e-10)

        np.testing.assert_allclose(
            float(fn.loss_val()), float(f1.loss_val()), rtol=RTOL
        )

        p = tf.constant(rng.standard_normal(f1.x.shape[0]))
        _, _, h1 = f1.loss_val_grad_hessp(p)
        _, _, hn = fn.loss_val_grad_hessp(p)
        np.testing.assert_allclose(hn.numpy(), h1.numpy(), rtol=1e-9, atol=1e-9)

        _, _, H1 = f1.loss_val_grad_hess()
        _, _, Hn = fn.loss_val_grad_hess()
        np.testing.assert_allclose(Hn.numpy(), H1.numpy(), rtol=1e-9, atol=1e-9)


def test_sharded_profile_beta_matches():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = make_test_tensor(tmpdir)
        f1 = _make_fitter(fname, 1)
        fn = _make_fitter(fname, 2)

        rng = np.random.default_rng(7)
        xval = f1.x.numpy() + 0.05 * rng.standard_normal(f1.x.shape[0])
        f1.x.assign(xval)
        fn.x.assign(xval)

        f1._profile_beta()
        fn._profile_beta()
        np.testing.assert_allclose(
            fn.bbstat.beta.numpy(), f1.bbstat.beta.numpy(), rtol=1e-10, atol=1e-12
        )


@pytest.mark.parametrize("method", ["trust-krylov", "tf-trust-krylov"])
def test_sharded_fit_matches_single_device(method):
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = make_test_tensor(tmpdir)
        f1 = _make_fitter(fname, 1, minimizerMethod=method)
        fn = _make_fitter(fname, 2, minimizerMethod=method)

        f1.minimize()
        fn.minimize()

        np.testing.assert_allclose(fn.x.numpy(), f1.x.numpy(), rtol=1e-6, atol=1e-7)
        v1, g1, H1 = f1.loss_val_grad_hess()
        vn, gn, Hn = fn.loss_val_grad_hess()
        # the offset-form NLL is ~0 at the minimum of this Asimov-like fit,
        # so compare absolutely at float64 cancellation scale
        np.testing.assert_allclose(float(vn), float(v1), atol=1e-9)

        from rabbit.tfhelpers import edmval_cov

        edm1, _ = edmval_cov(g1, H1)
        edmn, _ = edmval_cov(gn, Hn)
        assert float(edmn) < 1e-4 and float(edm1) < 1e-4


def test_sharded_rejects_unsupported():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = make_test_tensor(tmpdir, sparse=True)
        indata_obj = inputdata.FitInputData(fname)
        param_model = load_model("Mu", indata_obj)
        options = make_options(nDevices=2)
        with pytest.raises(NotImplementedError):
            fitter.make_fitter(indata_obj, param_model, options)


def test_sharded_fitter_deepcopy():
    """Toys deepcopy the fitter; the shard machinery must be rebuilt."""
    import copy

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = make_test_tensor(tmpdir)
        fn = _make_fitter(fname, 2)
        v0 = float(fn.loss_val())
        fc = copy.deepcopy(fn)
        assert len(fc.shards) == 2
        np.testing.assert_allclose(float(fc.loss_val()), v0, rtol=RTOL)


def test_sharded_across_logical_cpu_devices():
    """Run the sharded path across two genuinely distinct (logical CPU)
    devices in a subprocess, where the TF context can still be configured.

    This is the regression net for the bug class the same-device CPU tests
    are structurally blind to: XLA-compiled shard functions capturing
    state resident on another device (first seen as jit functions reading
    the fitter's frozen_params_mask Variable from GPU:0)."""
    import os
    import subprocess
    import sys
    import textwrap

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = make_test_tensor(tmpdir)
        script = textwrap.dedent(f"""
            import numpy as np
            import tensorflow as tf

            cpu = tf.config.list_physical_devices("CPU")[0]
            tf.config.set_logical_device_configuration(
                cpu,
                [tf.config.LogicalDeviceConfiguration()] * 2,
            )

            from rabbit import fitter, inputdata
            from rabbit.param_models.helpers import load_model
            from tests.test_sparse_fit import make_options

            def build(nd):
                indata = inputdata.FitInputData({fname!r}, host_memory=nd > 1)
                pm = load_model("Mu", indata)
                f = fitter.make_fitter(indata, pm, make_options(nDevices=nd))
                f.set_nobs(indata.data_obs)
                return f

            f1 = build(1)
            f2 = build(2)
            devs = {{s.device for s in f2.shards}}
            assert len(devs) == 2, devs

            rng = np.random.default_rng(5)
            xval = f1.x.numpy() + 0.1 * rng.standard_normal(f1.x.shape[0])
            f1.x.assign(xval)
            f2.x.assign(xval)

            v1, g1 = f1.loss_val_grad()
            v2, g2 = f2.loss_val_grad()
            np.testing.assert_allclose(float(v2), float(v1), rtol=1e-12)
            np.testing.assert_allclose(g2.numpy(), g1.numpy(), rtol=1e-10, atol=1e-10)

            p = tf.constant(rng.standard_normal(f1.x.shape[0]))
            _, _, h1 = f1.loss_val_grad_hessp(p)
            _, _, h2 = f2.loss_val_grad_hessp(p)
            np.testing.assert_allclose(h2.numpy(), h1.numpy(), rtol=1e-9, atol=1e-9)

            _, _, H1 = f1.loss_val_grad_hess()
            _, _, H2 = f2.loss_val_grad_hess()
            np.testing.assert_allclose(H2.numpy(), H1.numpy(), rtol=1e-9, atol=1e-9)

            f2._profile_beta()
            print("LOGICAL-DEVICE SHARDING OK")
            """)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [os.getcwd()] + env.get("PYTHONPATH", "").split(os.pathsep)
        )
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"
        res = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=420,
        )
        assert res.returncode == 0, res.stdout + "\n" + res.stderr
        assert "LOGICAL-DEVICE SHARDING OK" in res.stdout
