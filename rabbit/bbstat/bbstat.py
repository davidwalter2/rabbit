"""Bin-by-bin (BBB) statistical treatment for the rabbit fitter.

Owns all β-related state (TF Variables and precomputed tensors) and
exposes a small API used by :class:`rabbit.fitter.Fitter`. See the
package docstring in :mod:`rabbit.bbstat` for the high-level layout.
"""

import tensorflow as tf
from wums import logging

logger = logging.child_logger(__name__)


VALID_BIN_BY_BIN_STAT_TYPES = ["gamma", "normal-additive", "normal-multiplicative"]


class BinByBinStat:
    """Encapsulates bin-by-bin statistical treatment.

    Constructed once by :class:`rabbit.fitter.Fitter`. Owns β/β0/log β0/uβ
    TF Variables, the per-bin (or per-(bin, proc)) ``kstat`` / ``betamask``
    / ``proc_data_driven_mask`` tensors, and any precomputed LU
    decompositions used by the covarianceFit profile.

    The flag :attr:`enabled` mirrors ``not options.noBinByBinStat`` and
    callers should branch on it instead of the legacy
    ``Fitter.binByBinStat``.
    """

    valid_types = VALID_BIN_BY_BIN_STAT_TYPES

    def __init__(
        self,
        indata,
        options,
        *,
        chisqFit,
        covarianceFit,
        data_cov_inv,
        nobs_template,
    ):
        """
        Parameters
        ----------
        indata : FitInputData
            Loaded input tensor; only ``sumw``, ``sumw2``, ``nbins`` and
            ``dtype`` are read.
        options : argparse.Namespace
            CLI/options namespace with ``noBinByBinStat``, ``binByBinStatMode``,
            ``binByBinStatType``, and (optional) ``minBBKstat``.
        chisqFit, covarianceFit : bool
            Likelihood form selected on the Fitter.
        data_cov_inv : tf.Tensor or None
            Inverse data covariance matrix (only used by covarianceFit +
            normal-additive to precompute the auxiliary LU decomposition).
        nobs_template : tf.Tensor
            Template tensor with the same shape and dtype as
            ``Fitter.nobs``; used to size the per-bin ``nbeta`` Variable
            in the gamma + full mode.
        """
        self.indata = indata
        self.dtype = indata.dtype

        self.enabled = not options.noBinByBinStat
        self.binByBinStatMode = options.binByBinStatMode
        self.minBBKstat = getattr(options, "minBBKstat", 0.0)

        if options.binByBinStatType == "automatic":
            if covarianceFit:
                self.binByBinStatType = "normal-additive"
            elif options.binByBinStatMode == "full":
                self.binByBinStatType = "normal-multiplicative"
            else:
                self.binByBinStatType = "gamma"
        else:
            self.binByBinStatType = options.binByBinStatType

        if (
            covarianceFit
            and self.enabled
            and not self.binByBinStatType.startswith("normal")
        ):
            raise Exception(
                'bin-by-bin stat only for option "--covarianceFit" with '
                '"--binByBinStatType normal"'
            )

        if self.binByBinStatType not in VALID_BIN_BY_BIN_STAT_TYPES:
            raise RuntimeError(
                f"Invalid binByBinStatType {self.binByBinStatType}, valid "
                f"choices are {VALID_BIN_BY_BIN_STAT_TYPES}"
            )

        # FIXME for now this is needed even if binByBinStat is off because
        # of how it is used in the global impacts and uncertainty band
        # computations (gradient is allowed to be zero or None and then
        # propagated or skipped only later).

        # Shape of the β tensor: per-bin in lite mode, per-(bin, proc) in full.
        if self.binByBinStatMode == "full":
            self.beta_shape = self.indata.sumw.shape
        elif self.binByBinStatMode == "lite":
            self.beta_shape = (self.indata.sumw.shape[0],)

        self.beta0 = tf.Variable(
            tf.zeros(self.beta_shape, dtype=self.dtype),
            trainable=False,
            name="beta0",
        )
        self.logbeta0 = tf.Variable(
            tf.zeros(self.beta_shape, dtype=self.dtype),
            trainable=False,
            name="logbeta0",
        )
        self.beta0_default_assign()

        # nuisance parameters for mc stat uncertainty
        self.beta = tf.Variable(self.beta0, trainable=False, name="beta")

        # dummy tensor to allow differentiation by β even when profiling
        self.ubeta = tf.zeros_like(self.beta)

        # Per-(bin, process) mask of data-driven entries (sumw2 == 0).
        # Used in lite mode to exclude these from the merged sumw reduction
        # and from the β scaling. Defaults to None (no per-process axis).
        self.proc_data_driven_mask = None

        # Optional: gamma+full uses a per-bin Newton iterate variable.
        self.nbeta = None
        # Optional: covarianceFit + normal-additive precomputes an LU.
        self.betaauxlu = None

        if self.enabled:
            if self.binByBinStatMode == "full":
                self.varbeta = self.indata.sumw2
                self.sumw = self.indata.sumw
            else:
                if self.indata.sumw2.ndim > 1:
                    # Identify data-driven processes per bin (no MC stat
                    # uncertainty). They are excluded from the merged sumw
                    # so the lite-mode kstat is computed from MC-only
                    # contributions.
                    self.proc_data_driven_mask = self.indata.sumw2 == 0.0
                    sumw_mc = tf.where(
                        self.proc_data_driven_mask,
                        tf.zeros_like(self.indata.sumw),
                        self.indata.sumw,
                    )
                    self.varbeta = tf.reduce_sum(self.indata.sumw2, axis=-1)
                    self.sumw = tf.reduce_sum(sumw_mc, axis=-1)
                else:
                    self.varbeta = self.indata.sumw2
                    self.sumw = self.indata.sumw

            self.betamask = (self.varbeta == 0.0) | (self.sumw == 0.0)
            if self.minBBKstat > 0.0:
                # Mask (bin, process) entries with effective MC stats below
                # threshold to avoid ill-conditioned profiles (e.g. from
                # mixed-sign-weight cancellations).
                varbeta_safe = tf.where(
                    self.betamask,
                    tf.ones_like(self.varbeta),
                    self.varbeta,
                )
                kstat_eff = self.sumw**2 / varbeta_safe
                low_stat = kstat_eff < tf.constant(
                    self.minBBKstat, dtype=self.varbeta.dtype
                )
                n_extra = int(
                    tf.reduce_sum(tf.cast(low_stat & ~self.betamask, tf.int32)).numpy()
                )
                if n_extra > 0:
                    logger.info(
                        f"--minBBKstat {self.minBBKstat}: masking {n_extra} "
                        "additional low-stat (bin, process) entries"
                    )
                self.betamask = self.betamask | low_stat
            self.kstat = tf.where(self.betamask, 1.0, self.sumw**2 / self.varbeta)

            if self.binByBinStatType in ["gamma", "normal-multiplicative"]:
                if self.binByBinStatType == "gamma" and self.binByBinStatMode == "full":
                    logger.warning(
                        "Running with '--binByBinStatType gamma "
                        "--binByBinStatMode full' is experimental and "
                        "results should be taken with care"
                    )
                    self.nbeta = tf.Variable(
                        tf.ones_like(nobs_template), trainable=True, name="nbeta"
                    )

            elif self.binByBinStatType == "normal-additive":
                # precompute decomposition of composite matrix to speed up
                # calculation of profiled beta values
                if covarianceFit:
                    sbeta = tf.math.sqrt(self.varbeta[: self.indata.nbins])

                    if self.binByBinStatMode == "lite":
                        sbeta = tf.linalg.LinearOperatorDiag(sbeta)
                        self.betaauxlu = tf.linalg.lu(
                            sbeta @ data_cov_inv @ sbeta
                            + tf.eye(
                                data_cov_inv.shape[0],
                                dtype=data_cov_inv.dtype,
                            )
                        )
                    elif self.binByBinStatMode == "full":
                        varbetasum = tf.reduce_sum(
                            self.varbeta[: self.indata.nbins], axis=1
                        )
                        varbetasum = tf.linalg.LinearOperatorDiag(varbetasum)
                        self.betaauxlu = tf.linalg.lu(
                            varbetasum @ data_cov_inv
                            + tf.eye(
                                data_cov_inv.shape[0],
                                dtype=data_cov_inv.dtype,
                            )
                        )

    # --- helpers -----------------------------------------------------------

    def default_beta0(self):
        """Default β0 tensor: 1 for multiplicative constraints, 0 for additive."""
        if self.binByBinStatType in ["gamma", "normal-multiplicative"]:
            return tf.ones(self.beta_shape, dtype=self.dtype)
        elif self.binByBinStatType == "normal-additive":
            return tf.zeros(self.beta_shape, dtype=self.dtype)

    def set_beta0(self, values):
        """Assign β0 and update the cached log β0 used in the gamma constraint."""
        self.beta0.assign(values)
        # Compute offset for Gamma NLL improved numerical precision in the
        # minimizer; the offset is chosen to give the saturated likelihood.
        beta0safe = tf.where(
            values == 0.0, tf.constant(1.0, dtype=values.dtype), values
        )
        self.logbeta0.assign(tf.math.log(beta0safe))

    def beta0_default_assign(self):
        self.set_beta0(self.default_beta0())

    def beta_default_assign(self):
        self.beta.assign(self.beta0)
