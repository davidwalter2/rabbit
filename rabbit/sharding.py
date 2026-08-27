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

from rabbit import external_likelihood
from rabbit.bbstat.bbstat import BinByBinStat
from rabbit.fitter import Fitter

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


class MultiDeviceFitter(Fitter):
    """Bins-sharded Fitter for multi-device (typically multi-GPU) fits.

    The device layout is fixed at construction (use
    :func:`rabbit.fitter.make_fitter` to pick the class), so all sharding
    lives here and the base Fitter carries none of it. The likelihood
    mathematics is not reimplemented: shard evaluators borrow the base
    class's own unbound methods (see _SHARD_BORROWED_METHODS), so this
    class owns orchestration only -- building the per-device shards and
    assembling loss/grad/HVP/Hessian from per-shard partials.

    ``_make_tf_functions`` (called at the end of init_fit_parms and after
    a deepcopy) rebuilds the shards first, so parameter-layout changes and
    toy copies stay consistent without extra hooks in the base class.
    """

    _DYNAMIC_TF_FUNCS = Fitter._DYNAMIC_TF_FUNCS | {
        "shards",
        "_shard_graph_fns",
        "_global_view",
        "_global_graph_fns",
    }

    def __init__(self, indata, param_model, options, do_blinding=False):
        # set before super().__init__: the base constructor ends in
        # init_fit_parms -> _make_tf_functions, which builds the shards
        self.n_devices = int(getattr(options, "nDevices", 1) or 1)
        self._options = options
        self.shards = []
        super().__init__(indata, param_model, options, do_blinding=do_blinding)

    _SHARD_BORROWED_METHODS = (
        "get_poi",
        "get_model_nui",
        "get_theta",
        "get_x",
        "_compute_yields_noBBB",
        "_compute_yields_with_beta",
        "_compute_ln",
        "_compute_lbeta",
        "_compute_lc",
    )

    def _build_shards(self):
        """Construct the per-device shard evaluators (multi-device mode).

        Shards partition the observed bins contiguously; each holds its
        slice of logk / norm / nobs / sumw(2) and its own BinByBinStat
        (beta and kstat are per-bin, so BBB shards along with everything
        else). Parameter-level state (x, frozen mask, blinding offsets,
        param model) is shared -- those tensors are [nparams]-sized and
        cross devices for kilobytes per evaluation.
        """
        if self.indata.sparse:
            raise NotImplementedError(
                "Multi-device fits (--nDevices > 1) are not supported in "
                "sparse mode."
            )
        if self.covarianceFit:
            raise NotImplementedError(
                "Multi-device fits are not supported with --covarianceFit "
                "(the dense data covariance couples all bins)."
            )

        devices = select_devices(self.n_devices)
        edges = shard_edges(self.indata.nbins, self.n_devices)
        logger.info(
            f"Sharding {self.indata.nbins} bins over {self.n_devices} "
            f"device(s): {[f'{d}:[{a},{b})' for d, (a, b) in zip(devices, edges)]}"
        )

        self.shards = []
        for device, (a, b) in zip(devices, edges):
            view = ShardIndataView(self.indata, a, b, device)
            shard = ShardEvaluator(device, view)
            with tf.device(device):
                shard.logk = tf.identity(self.logk[a:b])
                shard.nobs = tf.Variable(
                    tf.zeros([b - a], dtype=self.indata.dtype),
                    trainable=False,
                    name=f"nobs_shard{a}",
                )
                shard.lognobs = tf.Variable(
                    tf.zeros([b - a], dtype=self.indata.dtype),
                    trainable=False,
                    name=f"lognobs_shard{a}",
                )
                shard.varnobs = (
                    tf.Variable(
                        tf.ones([b - a], dtype=self.indata.dtype),
                        trainable=False,
                        name=f"varnobs_shard{a}",
                    )
                    if self.chisqFit
                    else None
                )
                shard.bbstat = BinByBinStat(
                    view,
                    self._options,
                    chisqFit=self.chisqFit,
                    covarianceFit=False,
                    data_cov_inv=None,
                    nobs_template=shard.nobs,
                )
            # seed from the current observation state: init_fit_parms (and
            # thus shard construction) can re-run after set_nobs
            shard.nobs.assign(self.nobs[a:b])
            shard.lognobs.assign(self.lognobs[a:b])
            if shard.varnobs is not None:
                shard.varnobs.assign(self.varnobs[a:b])
            shard.param_model = ShardParamModel(self.param_model, a, b)
            # Parameter-level state. NB the fitter's tf.Variables
            # (frozen_params_mask, blinding offsets) must NOT be captured
            # here: XLA-compiled functions cannot read a Variable resident
            # on a different device, so their *values* are threaded into the
            # per-shard functions as arguments instead (see
            # _make_sharded_tf_functions), the same way x is.
            shard.do_blinding = self.do_blinding
            shard.chisqFit = self.chisqFit
            shard.covarianceFit = False
            shard.data_cov_inv = None
            for name in self._SHARD_BORROWED_METHODS:
                setattr(shard, name, getattr(Fitter, name).__get__(shard))
            self.shards.append(shard)

        # the parameter-level NLL terms (constraints + external likelihood
        # terms), evaluated once per call on the first device via the same
        # borrowed-method pattern
        gview = ShardEvaluator(devices[0], self.indata)
        gview.param_model = self.param_model
        gview.frozen_params_mask = self.frozen_params_mask
        gview.do_blinding = self.do_blinding
        if self.do_blinding:
            gview._blinding_offsets_poi = self._blinding_offsets_poi
            gview._blinding_offsets_theta = self._blinding_offsets_theta
        gview.cw = self.cw
        gview.x0 = self.x0
        for name in ("get_poi", "get_model_nui", "get_theta", "get_x", "_compute_lc"):
            setattr(gview, name, getattr(Fitter, name).__get__(gview))
        self._global_view = gview

    def set_nobs(self, values, variances=None):
        super().set_nobs(values, variances)
        nobssafe = tf.where(values == 0.0, tf.constant(1.0, dtype=values.dtype), values)
        for shard in self.shards:
            a, b = shard.indata.start, shard.indata.stop
            shard.nobs.assign(values[a:b])
            shard.lognobs.assign(tf.math.log(nobssafe[a:b]))
            if shard.varnobs is not None:
                shard.varnobs.assign(self.varnobs[a:b])

    def _make_tf_functions(self):
        """Multi-device counterparts of the loss/grad/HVP/Hessian wrappers.

        Rebuilds the shards first: this method runs at the end of
        init_fit_parms and after a deepcopy, exactly the moments the shard
        evaluators must be reconstructed against the current parameter
        layout.

        The per-shard functions are jit-compiled on their own device and
        differentiate with respect to their local x copy (one tape per
        shard); the combiners below are plain graphs that broadcast x,
        collect the [nparams]-sized partials and sum them. See
        rabbit/sharding.py for why both choices are load-bearing.
        """
        self._build_shards()

        jit = self.jit_compile

        def make_shard_fns(shard):
            # aux = (frozen_params_mask value[, blinding offsets]) -- Variable
            # values threaded in as tensors because XLA cannot read a
            # Variable on another device
            def _pin(x, aux):
                shard.x = x
                shard.frozen_params_mask = aux[0]
                if shard.do_blinding:
                    shard._blinding_offsets_poi = aux[1]
                    shard._blinding_offsets_theta = aux[2]

            def nll_local(x, aux):
                _pin(x, aux)
                nexp, _, beta = shard._compute_yields_with_beta(
                    profile=True, compute_norm=False, full=False
                )
                ln = shard._compute_ln(nexp[: shard.indata.nbins], full_nll=False)
                lbeta = shard._compute_lbeta(beta, full_nll=False)
                return ln + lbeta if lbeta is not None else ln

            def vg(x, aux):
                with tf.GradientTape() as t:
                    t.watch(x)
                    v = nll_local(x, aux)
                g = t.gradient(v, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                return v, g

            def vgp(x, aux, p):
                p = tf.stop_gradient(p)
                with tf.GradientTape() as t2:
                    t2.watch(x)
                    with tf.GradientTape() as t1:
                        t1.watch(x)
                        v = nll_local(x, aux)
                    g = t1.gradient(
                        v, x, unconnected_gradients=tf.UnconnectedGradients.ZERO
                    )
                hp = t2.gradient(
                    g,
                    x,
                    output_gradients=p,
                    unconnected_gradients=tf.UnconnectedGradients.ZERO,
                )
                return v, g, hp

            def vgh(x, aux):
                with tf.GradientTape() as t2:
                    t2.watch(x)
                    with tf.GradientTape() as t1:
                        t1.watch(x)
                        v = nll_local(x, aux)
                    g = t1.gradient(
                        v, x, unconnected_gradients=tf.UnconnectedGradients.ZERO
                    )
                h = t2.jacobian(g, x)
                return v, g, h

            def beta_local(x, aux):
                _pin(x, aux)
                _, _, beta = shard._compute_yields_with_beta(
                    profile=True, compute_norm=False, full=False
                )
                return beta

            return (
                tf.function(nll_local, jit_compile=jit),
                tf.function(vg, jit_compile=jit),
                tf.function(vgp, jit_compile=jit),
                tf.function(vgh, jit_compile=jit),
                tf.function(beta_local, jit_compile=jit),
            )

        self._shard_graph_fns = [make_shard_fns(shard) for shard in self.shards]

        gview = self._global_view

        def gnll_local(x):
            gview.x = x
            lc = gview._compute_lc(full_nll=False)
            lext = external_likelihood.compute_external_nll(
                self.external_terms, x, self.indata.dtype, full_nll=False
            )
            return lc + lext if lext is not None else lc

        def gvg(x):
            with tf.GradientTape() as t:
                t.watch(x)
                v = gnll_local(x)
            g = t.gradient(v, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            return v, g

        def gvgp(x, p):
            p = tf.stop_gradient(p)
            with tf.GradientTape() as t2:
                t2.watch(x)
                with tf.GradientTape() as t1:
                    t1.watch(x)
                    v = gnll_local(x)
                g = t1.gradient(
                    v, x, unconnected_gradients=tf.UnconnectedGradients.ZERO
                )
            hp = t2.gradient(
                g,
                x,
                output_gradients=p,
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )
            return v, g, hp

        def gvgh(x):
            with tf.GradientTape() as t2:
                t2.watch(x)
                with tf.GradientTape() as t1:
                    t1.watch(x)
                    v = gnll_local(x)
                g = t1.gradient(
                    v, x, unconnected_gradients=tf.UnconnectedGradients.ZERO
                )
            h = t2.jacobian(g, x)
            return v, g, h

        self._global_graph_fns = tuple(
            tf.function(f, jit_compile=jit) for f in (gnll_local, gvg, gvgp, gvgh)
        )

        def _read_aux():
            aux = [tf.identity(self.frozen_params_mask)]
            if self.do_blinding:
                aux.append(tf.identity(self._blinding_offsets_poi))
                aux.append(tf.identity(self._blinding_offsets_theta))
            return tuple(aux)

        def _to_device(aux):
            return tuple(tf.identity(a) for a in aux)

        def _loss_val():
            x0 = tf.identity(self.x)
            aux = _read_aux()
            parts = []
            for shard, fns in zip(self.shards, self._shard_graph_fns):
                with tf.device(shard.device):
                    parts.append(fns[0](tf.identity(x0), _to_device(aux)))
            with tf.device(self.shards[0].device):
                parts.append(self._global_graph_fns[0](tf.identity(x0)))
            return tf.add_n(parts)

        def _loss_val_grad():
            x0 = tf.identity(self.x)
            aux = _read_aux()
            vs, gs = [], []
            for shard, fns in zip(self.shards, self._shard_graph_fns):
                with tf.device(shard.device):
                    v, g = fns[1](tf.identity(x0), _to_device(aux))
                vs.append(v)
                gs.append(g)
            with tf.device(self.shards[0].device):
                v, g = self._global_graph_fns[1](tf.identity(x0))
            vs.append(v)
            gs.append(g)
            return tf.add_n(vs), tf.add_n(gs)

        def _loss_val_grad_hessp(p):
            x0 = tf.identity(self.x)
            aux = _read_aux()
            vs, gs, hs = [], [], []
            for shard, fns in zip(self.shards, self._shard_graph_fns):
                with tf.device(shard.device):
                    v, g, hp = fns[2](tf.identity(x0), _to_device(aux), tf.identity(p))
                vs.append(v)
                gs.append(g)
                hs.append(hp)
            with tf.device(self.shards[0].device):
                v, g, hp = self._global_graph_fns[2](tf.identity(x0), tf.identity(p))
            vs.append(v)
            gs.append(g)
            hs.append(hp)
            return tf.add_n(vs), tf.add_n(gs), tf.add_n(hs)

        def _loss_val_grad_hess(profile=True):
            if not profile:
                raise NotImplementedError(
                    "profile=False Hessians are not supported in multi-device "
                    "mode (--nDevices > 1)."
                )
            x0 = tf.identity(self.x)
            aux = _read_aux()
            vs, gs, hs = [], [], []
            for shard, fns in zip(self.shards, self._shard_graph_fns):
                with tf.device(shard.device):
                    v, g, h = fns[3](tf.identity(x0), _to_device(aux))
                vs.append(v)
                gs.append(g)
                hs.append(h)
            with tf.device(self.shards[0].device):
                v, g, h = self._global_graph_fns[3](tf.identity(x0))
            vs.append(v)
            gs.append(g)
            hs.append(h)
            return tf.add_n(vs), tf.add_n(gs), tf.add_n(hs)

        def _profile_beta_sharded():
            x0 = tf.identity(self.x)
            aux = _read_aux()
            betas = []
            for shard, fns in zip(self.shards, self._shard_graph_fns):
                with tf.device(shard.device):
                    beta = fns[4](tf.identity(x0), _to_device(aux))
                shard.bbstat.beta.assign(beta)
                betas.append(beta)
            # mirror into the fitter-level beta variable, whose masked-bin
            # tail (never touched by the sharded loss) keeps its prior value
            nbins = self.indata.nbins
            full = tf.concat(betas + [self.bbstat.beta[nbins:]], axis=0)
            self.bbstat.beta.assign(full)

        self.loss_val = tf.function(_loss_val)
        self.loss_val_grad = tf.function(_loss_val_grad)
        self.loss_val_grad_hessp_revrev = tf.function(_loss_val_grad_hessp)
        if self.hvp_method == "fwdrev":
            logger.warning(
                "fwdrev HVP is not supported in multi-device mode; "
                "falling back to revrev."
            )
        self.loss_val_grad_hessp_fwdrev = self.loss_val_grad_hessp_revrev
        self.loss_val_grad_hessp = self.loss_val_grad_hessp_revrev
        # instance override of the class-level @tf.function
        self.loss_val_grad_hess = _loss_val_grad_hess
        if self.bbstat.enabled:
            self._profile_beta = tf.function(_profile_beta_sharded)
