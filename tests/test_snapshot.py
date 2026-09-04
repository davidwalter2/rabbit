"""Snapshots: an interrupted fit must not be a lost fit.

The contract is narrow and worth stating: a snapshot has to be loadable by
Fitter.load_fitresult, which is what --externalPostfit calls. Anything that
writes a file load_fitresult will not read is useless however well formed.
"""

import os
import signal
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import pytest

from rabbit import fitter, inputdata
from rabbit.param_models.helpers import load_model
from rabbit.snapshot import Snapshotter, write_snapshot

from .test_sparse_fit import make_options, make_test_tensor


def test_snapshot_round_trips_through_load_fitresult():
    """The point of the whole feature: a snapshot resumes a fit.

    Loading is done by the same call --externalPostfit uses, so this fails if
    the file layout ever drifts from what load_fitresult accepts.
    """
    with tempfile.TemporaryDirectory() as tmp:
        filename = make_test_tensor(tmp)
        indata_obj = inputdata.FitInputData(filename)
        param_model = load_model("Mu", indata_obj)
        f = fitter.Fitter(indata_obj, param_model, make_options())
        f.set_nobs(indata_obj.data_obs)
        f.minimize()
        fitted = f.x.numpy().copy()

        snap = os.path.join(tmp, "snapshot.hdf5")
        write_snapshot(snap, f.parms, fitted)

        # a fresh fitter, deliberately somewhere else
        g = fitter.Fitter(indata_obj, load_model("Mu", indata_obj), make_options())
        g.set_nobs(indata_obj.data_obs)
        g.x.assign(np.zeros_like(fitted))
        g.load_fitresult(snap, None, profile=False)
        assert np.allclose(g.x.numpy(), fitted, rtol=0, atol=0)


def test_snapshot_undoes_the_preconditioner():
    """Internal coordinates would load without complaint and be wrong."""
    parms = np.array(["a", "b"])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.hdf5")
        s = Snapshotter(path, parms, to_physical=lambda v: 10.0 * v)
        s.save(np.array([1.0, 2.0]), "test")
        with h5py.File(path, "r") as h:
            assert np.allclose(h["x"][...], [10.0, 20.0])
            assert list(h["parms"][...].astype(str)) == ["a", "b"]


def test_conversion_uses_the_transform_current_at_the_time():
    """A rebuilt preconditioner must not retroactively remap an old iterate.

    The rebuild takes minutes on a real model, and a signal arriving inside
    that window would otherwise write a vector mapped by the wrong transform.
    """
    scale = [2.0]
    parms = np.array(["a"])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.hdf5")
        s = Snapshotter(path, parms, to_physical=lambda v: scale[0] * v)
        s.update(np.array([3.0]))  # recorded as 6.0
        scale[0] = 1000.0  # transform rebuilt
        s.save_latest("signal")
        with h5py.File(path, "r") as h:
            assert np.allclose(h["x"][...], [6.0])


def test_interrupted_write_cannot_destroy_the_previous_snapshot():
    """The failure mode the atomic write exists for."""
    parms = np.array(["a"])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.hdf5")
        write_snapshot(path, parms, np.array([1.0]))

        class Boom(Exception):
            pass

        def explode(*a, **k):
            raise Boom

        s = Snapshotter(path, parms)
        s.latest = np.array([2.0])
        import rabbit.snapshot as mod

        original, mod.write_snapshot = mod.write_snapshot, explode
        try:
            assert s._write("test") is False  # swallowed, not raised
        finally:
            mod.write_snapshot = original
        with h5py.File(path, "r") as h:
            assert np.allclose(h["x"][...], [1.0]), "old snapshot survived"
        assert [n for n in os.listdir(tmp) if n.startswith(".")] == []


def test_periodic_snapshots_respect_the_interval():
    parms = np.array(["a"])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.hdf5")
        s = Snapshotter(path, parms, interval_hours=1.0)
        x = np.array([1.0])
        assert s.maybe_save(x, 0.0) is True  # first write proves the path works
        assert s.maybe_save(x, 1800.0) is False  # half an hour later
        assert s.maybe_save(x, 3700.0) is True  # past the hour
        assert s.count == 2


def test_periodic_off_by_default_but_still_tracks_the_point():
    """interval 0 must not write, yet an interrupt still has a point to save."""
    parms = np.array(["a"])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.hdf5")
        s = Snapshotter(path, parms, interval_hours=0.0)
        assert s.maybe_save(np.array([7.0]), 1e9) is False
        assert not os.path.exists(path)
        assert s.save_latest("signal") is True
        with h5py.File(path, "r") as h:
            assert np.allclose(h["x"][...], [7.0])


def test_no_file_configured_is_a_silent_no_op():
    s = Snapshotter(None, np.array(["a"]))
    assert s.maybe_save(np.array([1.0]), 1e9) is False
    assert s.save(np.array([1.0]), "x") is False


def test_mismatched_lengths_are_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="against"):
            write_snapshot(
                os.path.join(tmp, "s.hdf5"), np.array(["a", "b"]), np.array([1.0])
            )


def test_sigterm_writes_a_snapshot_before_the_process_dies():
    """SIGTERM is how a scheduler announces a wall clock limit.

    Run in a subprocess: the handler re-raises the signal, so the process is
    meant to actually die, and the test is that the file is on disk afterwards.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.hdf5")
        # rabbit.snapshot deliberately does not import the fitter, so this
        # subprocess starts without TensorFlow and the test runs in seconds
        script = f"""
import os, signal, time, numpy as np
from rabbit.snapshot import Snapshotter, snapshot_on_signal
s = Snapshotter({path!r}, np.array(["a", "b"]))
s.update(np.array([1.5, -2.5]))
with snapshot_on_signal(s):
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(30)
"""
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, timeout=60
        )
        assert proc.returncode == -signal.SIGTERM, (
            f"expected death by SIGTERM, got {proc.returncode}: "
            f"{proc.stderr.decode()[-500:]}"
        )
        with h5py.File(path, "r") as h:
            assert np.allclose(h["x"][...], [1.5, -2.5])
            assert h.attrs["reason"] == "signal-SIGTERM"
