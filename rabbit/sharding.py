"""Multi-device (bins-sharded) evaluation of the likelihood.

The NLL is a plain sum over bins plus parameter-level terms, and every
large tensor in the fit (logk, norm, nobs, sumw/sumw2, beta) is
bins-proportional. Sharding the bins axis over N devices therefore
splits both the memory and the compute, while the only things that ever
cross devices per evaluation are the parameter vector going out and one
[nparams] partial (gradient / HVP) per shard coming back -- kilobytes
against the gigabytes each shard reads locally.

The two structural rules, established empirically (bench/poc_multigpu.py
on 4 GPUs):

1. Backward placement must be explicit. A single GradientTape over the
   combined multi-device loss computes correct values but the placer
   puts the backward ops on one device, dragging every shard's logk
   across the bus per call (measured 6.5 ms -> 135 ms at 4 shards).
   Hence: one tape per shard, differentiating with respect to a
   device-local copy of x, so each shard's backward subgraph is anchored
   to its device. The combined loss+grad returned to 4.5 ms.
2. XLA clusters are single-device: each shard's function is jit-compiled
   on its own, and the thin combiner stays a plain (non-jit) graph.

Rather than duplicating the yields/NLL mathematics, each shard is a
duck-typed evaluator object exposing the attributes the existing Fitter
methods read (indata view, logk slice, nobs, bbstat, param model, x);
the Fitter's own unbound methods are then bound onto it, so symmetric
and asymmetric interpolation, both systematic types, and the full
bin-by-bin-stat machinery are shared with the single-device path by
construction.

Not supported in sharded mode (checked at construction): sparse tensors,
--covarianceFit (dense cross-bin data covariance), regularizers (need
full=True yields), and the fwdrev HVP (falls back to revrev).
"""

import numpy as np
import tensorflow as tf
from wums import logging

logger = logging.child_logger(__name__)


def pick_physical_gpus(n, explicit=None):
    """Choose which physical GPUs to make visible, before TF initializes.

    ``explicit`` (a list of indices, from --devices) wins outright.
    Otherwise, when the node has more GPUs than requested, prefer the
    least-occupied ones by current memory use (via nvidia-smi), so a fit
    started on an interactively shared node lands next to nobody. Falls
    back to the first n on any query failure. NB nvidia-smi orders by PCI
    bus while CUDA defaults to fastest-first; on the homogeneous nodes
    here the orders coincide, and --devices remains the explicit escape
    hatch.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return None
    if explicit is not None:
        try:
            return [gpus[int(i)] for i in explicit]
        except IndexError:
            raise ValueError(
                f"--devices {explicit} out of range: {len(gpus)} GPU(s) visible"
            )
    if len(gpus) <= n:
        return gpus

    import subprocess

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        usage = {}
        for line in out.strip().splitlines():
            idx, used = line.split(",")
            usage[int(idx)] = int(used)
        order = sorted(range(len(gpus)), key=lambda i: usage.get(i, 0))
        chosen = sorted(order[:n])
        logger.info(
            f"Selecting GPU(s) {chosen} by occupancy " f"(memory used per GPU: {usage})"
        )
        return [gpus[i] for i in chosen]
    except Exception as ex:
        logger.warning(
            f"Could not query GPU occupancy ({ex}); using the first {n} GPU(s)."
        )
        return gpus[:n]


def select_devices(n):
    """Names of the n devices to shard over.

    Prefers distinct GPUs. With fewer GPUs than shards the shards are
    placed round-robin (still correct, still memory-split per device up
    to the number of GPUs); with no GPU at all every shard lands on the
    CPU -- functionally identical, which is what the correctness tests
    rely on.
    """
    gpus = tf.config.list_logical_devices("GPU")
    if len(gpus) >= n:
        return [g.name for g in gpus[:n]]
    if gpus:
        logger.warning(
            f"Requested {n} device shards but only {len(gpus)} GPU(s) are "
            "visible; placing shards round-robin."
        )
        return [gpus[i % len(gpus)].name for i in range(n)]
    cpus = tf.config.list_logical_devices("CPU")
    if len(cpus) >= n:
        # multiple logical CPU devices (tf.config.set_logical_device_
        # configuration): genuine cross-device placement without GPUs --
        # this is how the tests catch placement bugs XLA only raises
        # across devices, e.g. jit functions capturing a Variable resident
        # elsewhere
        return [c.name for c in cpus[:n]]
    logger.warning(
        f"Requested {n} device shards but no GPU is visible; running all "
        "shards on the CPU (correct, but no speedup)."
    )
    return [cpus[0].name] * n


class ShardIndataView:
    """The slice of FitInputData one shard sees, resident on its device.

    Only the attributes the yields/NLL/BBB code reads are provided. The
    view covers a contiguous range of *observed* bins; masked channels
    are excluded (the sharded path only serves the full=False likelihood
    evaluations), so nbins == nbinsfull and nbinsmasked == 0.
    """

    def __init__(self, indata, start, stop, device):
        self.start = int(start)
        self.stop = int(stop)
        nb = self.stop - self.start
        with tf.device(device):
            self.norm = tf.identity(indata.norm[self.start : self.stop])
            self.sumw = tf.identity(indata.sumw[self.start : self.stop])
            self.sumw2 = tf.identity(indata.sumw2[self.start : self.stop])
        self.nbins = nb
        self.nbinsfull = nb
        self.nbinsmasked = 0
        self.nproc = indata.nproc
        self.nsyst = indata.nsyst
        self.symmetric_tensor = indata.symmetric_tensor
        self.systematic_type = indata.systematic_type
        self.sparse = False
        self.dtype = indata.dtype


class ShardParamModel:
    """Delegating wrapper that slices bin-dependent process rates.

    param_model.compute returns a tensor broadcastable against
    [nbins, nproc]; when a model makes it genuinely bin-dependent
    (leading dimension > 1) the shard must see only its own rows.
    Everything else (npoi, npou, nparams, defaults, ...) passes through.
    """

    def __init__(self, inner, start, stop):
        self._inner = inner
        self._start = int(start)
        self._stop = int(stop)

    def compute(self, param, full=False):
        rnorm = self._inner.compute(param, full)
        nlead = rnorm.shape[0]
        if nlead is not None and nlead > 1:
            rnorm = rnorm[self._start : self._stop]
        return rnorm

    def __getattr__(self, name):
        return getattr(self._inner, name)


class ShardEvaluator:
    """Duck-typed per-shard stand-in for the Fitter in the yields/NLL code.

    Plain attribute container; the Fitter binds its own unbound methods
    onto instances (see Fitter._build_shards), and the per-shard traced
    functions set ``x`` to the device-local parameter copy before
    calling them.
    """

    def __init__(self, device, indata_view):
        self.device = device
        self.indata = indata_view
        self.x = None  # set inside each traced shard function


def shard_edges(nbins, n):
    """Contiguous, balanced bin ranges. Bins are the unit of both memory
    and flops in the dense tensors, so equal bin counts balance the load."""
    edges = np.linspace(0, nbins, n + 1).astype(np.int64)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n)]
