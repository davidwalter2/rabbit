"""Parameter snapshots, so an interrupted fit is not a lost fit.

A long fit is fragile in a way its length makes expensive. Everything the
minimiser has learned lives in one in-memory vector until the very end: the
output file is written only after ``minimize()`` returns, so a fit that is
killed, hits a wall clock limit, or dies writing its output leaves nothing at
all behind, however close to the minimum it had got.

A snapshot is that vector on disk. It is deliberately the smallest thing that
:meth:`rabbit.fitter.Fitter.load_fitresult` will accept -- the parameter values
and their names -- so a fit can be resumed from one with

    rabbit_fit.py input.hdf5 -o out/ --externalPostfit snapshot.hdf5

either to carry on minimising or, with ``--noFit``, to run just the postfit
step at the snapshot point. No covariance is stored: it does not exist until
the Hessian is computed, and ``load_fitresult`` treats it as optional.

Two things here are not incidental.

*The write is atomic.* Snapshots exist for the case where the process dies at a
moment it did not choose, and that includes while a snapshot is being written.
Writing in place would then leave a truncated file where a good one used to be,
turning the safety net into the thing that destroys the result. Each snapshot
is written to a temporary file in the same directory and moved into place with
``os.replace``, which is atomic on POSIX, so a reader sees either the previous
snapshot or the new one.

*The values are physical.* Under preconditioning the minimiser works in
internal coordinates and the callback's iterate is in those coordinates; the
transform has to be undone before writing. A snapshot of internal coordinates
would load without complaint and be silently wrong.

Nothing here imports the fitter, and so nothing here imports TensorFlow --
worth keeping that way. It makes the module usable from a signal handler and
from a plain subprocess, where pulling in TensorFlow costs minutes.
"""

import contextlib
import os
import signal

import h5py
import numpy as np
from wums import logging

logger = logging.child_logger(__name__)


def write_snapshot(filename, parms, x, meta=None):
    """Write parameter values and names to ``filename``, atomically.

    ``parms`` and ``x`` must be aligned. ``meta`` is an optional dict of
    scalars stored as HDF5 attributes; it is provenance only and nothing reads
    it back, so a snapshot stays loadable if its contents ever change.
    """
    x = np.asarray(x)
    parms = np.asarray(parms).astype(str)
    if x.shape != parms.shape:
        raise ValueError(
            f"snapshot: {x.size} parameter values against {parms.size} names"
        )

    directory = os.path.dirname(os.path.abspath(filename))
    # NB same directory as the target: os.replace is only atomic within a
    # filesystem, and /tmp is routinely a different one
    tmp = os.path.join(directory, f".{os.path.basename(filename)}.tmp")

    with h5py.File(tmp, "w") as f:
        f.create_dataset("x", data=x)
        f.create_dataset(
            "parms", data=parms.astype(object), dtype=h5py.special_dtype(vlen=str)
        )
        for key, value in (meta or {}).items():
            f.attrs[key] = value
    os.replace(tmp, filename)


class Snapshotter:
    """Decides when to snapshot and where to put it.

    Holds the mapping back to physical coordinates, so callers never have to
    remember to undo the preconditioner themselves.
    """

    def __init__(self, filename, parms, to_physical=None, interval_hours=0.0):
        self.filename = filename
        self.parms = parms
        self.to_physical = to_physical if to_physical is not None else (lambda v: v)
        # negative or zero disables the periodic snapshots; an explicit save
        # (a signal, a failing minimiser) is always honoured
        self.interval = float(interval_hours) * 3600.0
        self.last_write = None
        self.count = 0
        # most recent accepted iterate, already in physical coordinates
        self.latest = None

    def update(self, xval):
        """Record ``xval`` (internal coordinates) as the latest good point.

        Converting on the way in rather than on the way out is what makes a
        snapshot safe during a preconditioner rebuild: for the minutes that
        takes, the stored transform no longer matches the stored iterate, and
        anything converting lazily would write a vector mapped by the wrong
        one.
        """
        self.latest = np.asarray(self.to_physical(np.asarray(xval)))

    def _write(self, reason, **meta):
        """Never raises: the snapshot is insurance, not the product, so a
        failure here must not take down a fit that is otherwise fine."""
        if self.filename is None or self.latest is None:
            return False
        try:
            write_snapshot(
                self.filename, self.parms, self.latest, meta={"reason": reason, **meta}
            )
        except Exception as ex:  # pragma: no cover - defensive
            logger.warning(f"Could not write snapshot to {self.filename}: {ex}")
            return False
        self.count += 1
        logger.info(
            f"Wrote parameter snapshot ({reason}) to {self.filename}; resume with "
            f"--externalPostfit {self.filename}"
        )
        return True

    def save(self, xval, reason, **meta):
        """Snapshot ``xval`` unconditionally, whatever the interval says."""
        self.update(xval)
        return self._write(reason, **meta)

    def save_latest(self, reason, **meta):
        """Snapshot the last point handed to :meth:`update`.

        This is the signal path: it must not need an iterate passed in, since
        a signal can arrive at any point in an iteration -- and with iterations
        running to hours on a large fit, waiting for the next one is not a
        usable answer.
        """
        return self._write(reason, **meta)

    def maybe_save(self, xval, elapsed, reason="periodic", **meta):
        """Record the point, and snapshot it if the interval has passed.

        The first call always writes. That is deliberate: it proves the path is
        writable at iteration 1 rather than at hour 100, which is when it
        matters and far too late to learn otherwise.
        """
        self.update(xval)
        if self.filename is None or self.interval <= 0:
            return False
        if self.last_write is not None and elapsed - self.last_write < self.interval:
            return False
        self.last_write = elapsed
        return self._write(reason, elapsed=elapsed, **meta)


@contextlib.contextmanager
def snapshot_on_signal(snapshotter):
    """Write a snapshot when the job is asked to stop, then get out of the way.

    SIGTERM is how a batch scheduler announces a wall clock limit and how
    ``kill`` reaches a fit someone has decided to stop; SIGINT is Ctrl-C.
    Both otherwise destroy every parameter value the fit has found.

    The snapshot is written from inside the handler rather than by asking
    the minimiser to stop at the next iteration. On a large model a single
    iteration can run for hours, and a scheduler that has sent SIGTERM will
    follow it with SIGKILL long before then -- so a cooperative stop would
    arrive too late to be the safety net this is for.

    The previous handler is then restored and the signal re-raised, so the
    process still dies the way the sender intended and with the right exit
    status. Nothing about the fit's own control flow changes.
    """
    if snapshotter.filename is None:
        yield
        return

    previous = {}

    def handle(signum, frame):
        snapshotter.save_latest(f"signal-{signal.Signals(signum).name}")
        signal.signal(signum, previous.get(signum, signal.SIG_DFL))
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.signal(sig, handle)
        except ValueError:
            # signal handlers can only be installed from the main thread;
            # off the main thread the periodic and failure snapshots still
            # work, so carry on rather than refusing to fit
            logger.debug(f"Could not install a {sig!r} snapshot handler")
    try:
        yield
    finally:
        for sig, prev in previous.items():
            signal.signal(sig, prev)
