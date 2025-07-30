import hashlib
import re
import time
from functools import partial
from typing import NamedTuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from jax.scipy.special import gammaln
from wums import logging

from rabbit import jaxhelpers as jh

logger = logging.child_logger(__name__)


# container for static objects
class Config(NamedTuple):
    nsyst: int
    nproc: int
    npoi: int
    nbins: int
    nbinsmasked: int
    nbinsfull: int
    externalCovariance: bool
    chisqFit: bool
    allowNegativePOI: bool
    symmetric_tensor: bool
    binByBinStat: bool
    binByBinStatType: str
    do_blinding: bool
    systematic_type: str


def _compute_yields_noBBB(params, xval, cfg, compute_norm, full):
    # compute_norm: compute yields for each process, otherwise inclusive
    # full: compute yields inclduing masked channels
    npoi = cfg.npoi
    poi = get_blinded_poi(params, xval, cfg)
    theta = get_blinded_theta(params, xval, cfg)

    nsyst = cfg.nsyst
    nproc = cfg.nproc
    # dtype = params["dtype"]

    rnorm = jnp.concat(
        [poi, jnp.ones([nproc - poi.shape[0]], dtype=jnp.float64)],
        axis=0,
    )

    mrnorm = jnp.expand_dims(rnorm, -1)
    ernorm = jnp.reshape(rnorm, [1, -1])

    normcentral = None
    if cfg.symmetric_tensor:
        mthetaalpha = jnp.reshape(theta, [nsyst, 1])
    else:
        # interpolation for asymmetric log-normal
        twox = 2.0 * theta
        twox2 = twox * twox
        alpha = 0.125 * twox * (twox2 * (3.0 * twox2 - 10.0) + 15.0)
        alpha = jnp.clip(alpha, -1.0, 1.0)

        thetaalpha = theta * alpha

        mthetaalpha = jnp.stack([theta, thetaalpha], axis=0)  # now has shape [2,nsyst]
        mthetaalpha = jnp.reshape(mthetaalpha, [2 * nsyst, 1])

    if full or cfg.nbinsmasked == 0:
        nbins = cfg.nbinsfull
        logk = params["logk"]
        norm = params["norm"]
    else:
        nbins = cfg.nbins
        logk = params["logk"][:nbins]
        norm = params["norm"][:nbins]

    if cfg.symmetric_tensor:
        mlogk = jnp.reshape(
            logk,
            [nbins * nproc, nsyst],
        )
    else:
        mlogk = jnp.reshape(
            logk,
            [nbins * nproc, 2 * nsyst],
        )

    logsnorm = mlogk @ mthetaalpha
    logsnorm = jnp.reshape(logsnorm, [nbins, nproc])

    if cfg.systematic_type == "log_normal":
        snorm = jnp.exp(logsnorm)
        snormnorm = snorm * norm
        nexpcentral = snormnorm @ mrnorm
        nexpcentral = jnp.squeeze(nexpcentral, -1)
        if compute_norm:
            normcentral = ernorm * snormnorm
    elif cfg.systematic_type == "normal":
        normcentral = norm * ernorm + logsnorm
        nexpcentral = jnp.sum(normcentral, axis=-1)

    return nexpcentral, normcentral


def _compute_yields_with_beta(
    params, xval, cfg, profile=True, compute_norm=False, full=True
):
    nexp, norm = _compute_yields_noBBB(
        params,
        xval,
        cfg,
        compute_norm,
        full,
    )

    if cfg.binByBinStat:

        if profile:
            # analytic solution for profiled barlow-beeston lite parameters for each combination
            # of likelihood and uncertainty form

            nbins = cfg.nbins
            nexp_profile = nexp[:nbins]
            nobs = params["nobs"]
            beta0 = params["beta0"][:nbins]
            # denominator in Gaussian likelihood is treated as a constant when computing
            # global impacts for example
            nobs0 = jax.lax.stop_gradient(nobs)

            if cfg.chisqFit:
                if cfg.binByBinStatType == "gamma":
                    kstat = params["kstat"][:nbins]
                    betamask = params["betamask"][:nbins]

                    abeta = nexp_profile**2
                    bbeta = kstat * nobs0 - nexp_profile * nobs
                    cbeta = -kstat * nobs0 * beta0
                    beta = (
                        0.5
                        * (-bbeta + jnp.sqrt(bbeta**2 - 4.0 * abeta * cbeta))
                        / abeta
                    )
                    beta = jnp.where(betamask, beta0, beta)
                elif cfg.binByBinStatType == "normal":
                    varbeta = params["sumw2"][:nbins]
                    sbeta = jnp.sqrt(varbeta)
                    if cfg.externalCovariance:

                        P, L, U = params["betaauxlu"]
                        rhs = (
                            sbeta[:, None]
                            * (params["data_cov_inv"] @ (nobs - nexp_profile)[:, None])
                            + beta0[:, None]
                        )
                        rhs_perm = P @ rhs
                        y = jax.scipy.linalg.solve_triangular(L, rhs_perm, lower=True)
                        beta = jax.scipy.linalg.solve_triangular(U, y, lower=False)

                        beta = jnp.squeeze(beta, axis=-1)
                    else:
                        beta = (sbeta * (nobs - nexp_profile) + nobs0 * beta0) / (
                            nobs0 + varbeta
                        )
            else:
                if cfg.binByBinStatType == "gamma":
                    kstat = params["kstat"][:nbins]
                    betamask = params["betamask"][:nbins]

                    beta = (nobs + kstat * beta0) / (nexp_profile + kstat)
                    beta = jnp.where(betamask, beta0, beta)
                elif cfg.binByBinStatType == "normal":
                    varbeta = params["sumw2"][:nbins]
                    sbeta = jnp.sqrt(varbeta)
                    abeta = sbeta
                    abeta = jnp.where(varbeta == 0.0, 1.0, abeta)
                    bbeta = varbeta + nexp_profile - sbeta * beta0
                    cbeta = sbeta * (nexp_profile - nobs) - nexp_profile * beta0
                    beta = (
                        0.5
                        * (-bbeta + jnp.sqrt(bbeta**2 - 4.0 * abeta * cbeta))
                        / abeta
                    )
                    beta = jnp.where(varbeta == 0.0, beta0, beta)

            if cfg.nbinsmasked:
                beta = jnp.concat([beta, params["beta0"][nbins:]], axis=0)
        else:
            beta = params["beta"]

        # Add dummy tensor to allow convenient differentiation by beta even when profiling
        beta = beta + params["ubeta"]

        betasel = beta[: nexp.shape[0]]

        if cfg.binByBinStatType == "gamma":
            betamask = params["betamask"][: nexp.shape[0]]
            nexp = jnp.where(betamask, nexp, nexp * betasel)
            if compute_norm:
                norm = jnp.where(betamask[..., None], norm, betasel[..., None] * norm)
        elif cfg.binByBinStatType == "normal":
            varbeta = params["sumw2"][: nexp.shape[0]]
            sbeta = jnp.sqrt(varbeta)
            nexpnorm = nexp[..., None]
            nexp = nexp + sbeta * betasel
            if compute_norm:
                # distribute the change in yields proportionally across processes
                norm = norm + sbeta[..., None] * betasel[..., None] * norm / nexpnorm
    else:
        beta = None

    return nexp, norm, beta


def get_blinded_theta(params, xval, cfg):
    theta = xval[cfg.npoi :]
    if cfg.do_blinding:
        return theta + params["blinding_offsets_theta"]
    else:
        return theta


def get_blinded_poi(params, xval, cfg):
    xpoi = xval[: cfg.npoi]
    if cfg.allowNegativePOI:
        poi = xpoi
    else:
        poi = jnp.square(xpoi)
    if cfg.do_blinding:
        return poi * params["blinding_offsets_poi"]
    else:
        return poi


def _compute_lc(params, xval, cfg, full_nll):
    # constraints
    npoi = cfg.npoi
    theta = get_blinded_theta(params, xval, cfg)
    lc = params["constraintweights"] * 0.5 * jnp.square(theta - params["theta0"])
    if full_nll:
        # normalization factor for normal distribution: log(1/sqrt(2*pi)) = -0.9189385332046727
        lc = lc + 0.9189385332046727 * params["constraintweights"]

    return jnp.sum(lc)


def _compute_lbeta(params, beta, cfg, full_nll):
    if cfg.binByBinStat:
        beta0 = params["beta0"]
        if cfg.binByBinStatType == "gamma":
            kstat = params["kstat"]

            betasafe = jnp.where(beta0 == 0.0, 1.0, beta)
            logbeta = jnp.log(betasafe)

            if full_nll:
                # constant terms
                lgammaalpha = gammaln(kstat * beta0)
                alphalntheta = -kstat * beta0 * jnp.log(kstat)

                lbeta = (
                    -kstat * beta0 * logbeta + kstat * beta + lgammaalpha + alphalntheta
                )
            else:
                lbeta = -kstat * beta0 * (logbeta - params["logbeta0"]) + kstat * (
                    beta - beta0
                )
        elif cfg.binByBinStatType == "normal":
            lbeta = 0.5 * jnp.square(beta - beta0)

            if full_nll:
                sigma2 = params["sumw2"] / jnp.square(params["sumw"])

                # normalization factor for normal distribution: log(1/sqrt(2*pi)) = -0.9189385332046727
                lbeta = (
                    lbeta
                    + sigma2.size * 0.9189385332046727
                    + 0.5 * jnp.sum(jnp.log(sigma2))
                )

        return jnp.sum(lbeta)

    return 0.0


def _compute_nll_components(params, xval, cfg, full_nll, profile, compute_norm, full):
    nexpfullcentral, _, beta = _compute_yields_with_beta(
        params, xval, cfg, profile, compute_norm, full
    )

    nexp = nexpfullcentral

    nobs = params["nobs"]

    if cfg.chisqFit:
        if cfg.externalCovariance:
            # Solve the system without inverting
            residual = jnp.reshape(nobs - nexp, [-1, 1])  # chi2 residual
            ln = 0.5 * jnp.sum(residual.T @ params["data_cov_inv"] @ residual)
        else:
            # stop_gradient needed in denominator here because it should be considered
            # constant when evaluating global impacts from observed data
            ln = 0.5 * jnp.sum(
                (nexp - nobs) ** 2 / jax.lax.stop_gradient(nobs), axis=-1
            )
    else:
        nexpsafe = jnp.where(nobs == 0.0, 1.0, nexp)
        lognexp = jnp.log(nexpsafe)

        # poisson term
        if full_nll:
            ldatafac = gammaln(nobs + 1)
            ln = jnp.sum(-nobs * lognexp + nexp + ldatafac, axis=-1)
        else:
            # poisson w/o constant factorial part term and with offset to improve numerical precision
            ln = jnp.sum(-nobs * (lognexp - params["lognobs"]) + nexp - nobs, axis=-1)

    lc = _compute_lc(params, xval, cfg, full_nll)

    lbeta = _compute_lbeta(
        params,
        beta,
        cfg,
        full_nll,
    )

    return ln, lc, lbeta, beta


def _compute_nll(params, xval, cfg, full_nll, profile, compute_norm, full):
    ln, lc, lbeta, beta = _compute_nll_components(
        params, xval, cfg, full_nll, profile, compute_norm, full
    )
    l = ln + lc + lbeta

    return l


def _compute_loss(params, xval, cfg):
    l = _compute_nll(
        params, xval, cfg, full_nll=False, profile=True, compute_norm=False, full=False
    )
    return l


@partial(jax.jit, static_argnames=["cfg"])
def loss_val_grad(params, x, cfg):
    return jax.value_and_grad(_compute_loss, argnums=1)(params, x, cfg)


@partial(jax.jit, static_argnames=["cfg"])
def loss_hessp(params, xval, pval, cfg):
    # Compute Hessian-vector product using forward-mode AD
    def grad_fn(x):
        return jax.grad(_compute_loss, argnums=1)(params, x, cfg)

    _, hessp = jax.jvp(grad_fn, (xval,), (pval,))

    return hessp


@partial(jax.jit, static_argnames=["cfg"])
def loss_val_grad_hessp(params, x, p, cfg):
    # Function to compute scalar-valued loss
    def loss_fn(x_):
        return _compute_loss(params, x_, cfg)

    # Compute loss and gradient using reverse-mode
    val, grad = jax.value_and_grad(loss_fn)(x)

    # Compute Hessian-vector product using forward-over-reverse
    _, hessp = jax.jvp(jax.grad(loss_fn), (x,), (p,))

    return val, grad, hessp


@partial(jax.jit, static_argnames=["cfg"])
def loss_hess(params, x, cfg):
    return jax.hessian(_compute_loss, argnums=1)(params, x, cfg)


@partial(jax.jit, static_argnames=["cfg"])
def loss_val_grad_hess(params, xval, cfg):
    def loss_fn(x):
        return _compute_loss(params, x, cfg)

    # Compute value and gradient efficiently
    val, grad = jax.value_and_grad(loss_fn)(xval)

    # Compute hessian separately (only once needed)
    hess = jax.hessian(loss_fn)(xval)

    return val, grad, hess


@partial(jax.jit, static_argnames=["cfg"])
def full_nll(params, xval, cfg):
    return _compute_nll(
        params, xval, cfg, full_nll=True, profile=True, compute_norm=False, full=False
    )


@partial(jax.jit, static_argnames=["cfg"])
def reduced_nll(params, xval, cfg):
    return _compute_nll(
        params, xval, cfg, full_nll=False, profile=True, compute_norm=False, full=False
    )


class FitterCallback:
    def __init__(self, xv):
        self.iiter = 0
        self.xval = xv

    def __call__(self, intermediate_result):
        logger.debug(f"Iteration {self.iiter}: loss value {intermediate_result.fun}")
        if np.isnan(intermediate_result.fun):
            raise ValueError(f"Loss value is NaN at iteration {self.iiter}")
        self.xval = intermediate_result.x
        self.iiter += 1


class Fitter:
    def __init__(self, indata, options, do_blinding=False):
        self.n_grad = 0
        self.n_hvp = 0

        self.time_grad = 0
        self.time_hvp = 0
        self.time_grad_copy_1 = 0
        self.time_grad_copy_2 = 0
        self.time_hvp_copy_1 = 0
        self.time_hvp_copy_2 = 0

        self.indata = indata
        self.binByBinStat = not options.noBinByBinStat
        self.systgroupsfull = self.indata.systgroups.tolist()
        self.systgroupsfull.append("stat")
        if self.binByBinStat:
            self.systgroupsfull.append("binByBinStat")

        if options.binByBinStatType == "automatic":
            self.binByBinStatType = "normal" if options.externalCovariance else "gamma"
        else:
            self.binByBinStatType = options.binByBinStatType

        if options.externalCovariance and not options.chisqFit:
            raise Exception(
                'option "--externalCovariance" only works with "--chisqFit"'
            )
        if (
            options.externalCovariance
            and self.binByBinStat
            and self.binByBinStatType != "normal"
        ):
            raise Exception(
                'option "--binByBinStat" only for options "--externalCovariance" with "--binByBinStatType normal"'
            )

        if self.binByBinStatType not in ["gamma", "normal"]:
            raise RuntimeError(
                f"Invalid binByBinStatType {self.indata.binByBinStatType}, valid choices are 'gamma' or 'normal'"
            )

        if self.indata.systematic_type not in ["log_normal", "normal"]:
            raise RuntimeError(
                f"Invalid systematic_type {self.indata.systematic_type}, valid choices are 'log_normal' or 'normal'"
            )

        self.diagnostics = options.diagnostics
        self.minimizer_method = options.minimizerMethod

        self.chisqFit = options.chisqFit
        self.externalCovariance = options.externalCovariance
        self.prefitUnconstrainedNuisanceUncertainty = (
            options.prefitUnconstrainedNuisanceUncertainty
        )

        self.nsystgroupsfull = len(self.systgroupsfull)

        self.pois = []

        if options.POIMode == "mu":
            self.npoi = self.indata.nsignals
            poidefault = options.POIDefault * jnp.ones(
                [self.npoi], dtype=self.indata.dtype
            )
            for signal in self.indata.signals:
                self.pois.append(signal)
        elif options.POIMode == "none":
            self.npoi = 0
            poidefault = jnp.zeros([], dtype=self.indata.dtype)
        else:
            raise Exception("unsupported POIMode")

        self.do_blinding = do_blinding
        if self.do_blinding:
            self._blinding_offsets_poi = jnp.ones([self.npoi], dtype=self.indata.dtype)
            self._blinding_offsets_theta = jnp.zeros(
                [self.indata.nsyst], dtype=self.indata.dtype
            )
            self.init_blinding_values(options.unblind)

        self.parms = np.concatenate([self.pois, self.indata.systs])

        self.allowNegativePOI = options.allowNegativePOI

        if self.allowNegativePOI:
            self.xpoidefault = poidefault
        else:
            self.xpoidefault = jnp.sqrt(poidefault)

        # tf variable containing all fit parameters
        thetadefault = jnp.zeros([self.indata.nsyst], dtype=self.indata.dtype)
        if self.npoi > 0:
            xdefault = jnp.concat([self.xpoidefault, thetadefault], axis=0)
        else:
            xdefault = thetadefault

        self.x = xdefault

        # observed number of events per bin
        self.nobs = jnp.zeros_like(self.indata.data_obs)
        self.lognobs = jnp.zeros_like(self.indata.data_obs)
        self.data_cov_inv = None

        if self.chisqFit:
            if self.externalCovariance:
                if self.indata.data_cov_inv is None:
                    raise RuntimeError("No external covariance found in input data.")
                # provided covariance
                self.data_cov_inv = self.indata.data_cov_inv

        # constraint minima for nuisance parameters
        self.theta0 = jnp.zeros([self.indata.nsyst], dtype=self.indata.dtype)

        # FIXME for now this is needed even if binByBinStat is off because of how it is used in the global impacts
        #  and uncertainty band computations (gradient is allowed to be zero or None and then propagated or skipped only later)

        # global observables for mc stat uncertainty
        self.beta0 = jnp.zeros_like(self.indata.sumw)
        self.logbeta0 = jnp.zeros_like(self.indata.sumw)
        self.set_beta0(self._default_beta0())

        # nuisance parameters for mc stat uncertainty
        self.beta = self.beta0

        # dummy tensor to allow differentiation
        self.ubeta = jnp.zeros_like(self.beta)

        # static objects that affect control flow and that don't change once instanciated
        self.static_params = Config(
            nsyst=self.indata.nsyst,
            nproc=self.indata.nproc,
            nbins=self.indata.nbins,
            nbinsmasked=self.indata.nbinsmasked,
            nbinsfull=self.indata.nbinsfull,
            npoi=self.npoi,
            externalCovariance=self.externalCovariance,
            chisqFit=self.chisqFit,
            allowNegativePOI=self.allowNegativePOI,
            symmetric_tensor=self.indata.symmetric_tensor,
            binByBinStat=self.binByBinStat,
            binByBinStatType=self.binByBinStatType,
            do_blinding=self.do_blinding,
            systematic_type=self.indata.systematic_type,
        )

        self.params = {
            "constraintweights": self.indata.constraintweights,
            "sumw": self.indata.sumw,
            "sumw2": self.indata.sumw2,
            "logk": self.indata.logk,
            "norm": self.indata.norm,
            "blinding_offsets_theta": self._blinding_offsets_theta,
            "blinding_offsets_poi": self._blinding_offsets_poi,
            "data_cov_inv": self.data_cov_inv,
            "theta0": self.theta0,
            "beta0": self.beta0,
            "beta": self.beta,
            "ubeta": self.ubeta,
            "logbeta0": self.logbeta0,
        }

        if self.binByBinStat:
            if jnp.any(self.indata.sumw2 < 0.0):
                raise ValueError("Negative variance for binByBinStat")

            if self.binByBinStatType == "gamma":
                self.kstat = self.indata.sumw**2 / self.indata.sumw2
                self.betamask = self.indata.sumw2 == 0.0
                self.kstat = jnp.where(self.betamask, 1.0, self.kstat)

                self.params["betamask"] = self.betamask
                self.params["kstat"] = self.kstat
            elif self.binByBinStatType == "normal" and self.externalCovariance:
                # precompute decomposition of composite matrix to speed up
                # calculation of profiled beta values
                varbeta = self.indata.sumw2[: self.indata.nbins]
                sbeta = jnp.sqrt(varbeta)
                A = sbeta[:, None] * self.data_cov_inv * sbeta[None, :] + jnp.eye(
                    self.data_cov_inv.shape[0], dtype=self.data_cov_inv.dtype
                )
                P, L, U = jax.scipy.linalg.lu(A)
                self.params["betaauxlu"] = (P, L, U)

        self.nexpnom = self.expected_yield()
        # parameter covariance matrix
        self.cov = self.prefit_covariance(
            unconstrained_err=self.prefitUnconstrainedNuisanceUncertainty
        )

        # determine if problem is linear (ie likelihood is purely quadratic)
        self.is_linear = (
            self.chisqFit
            and (self.npoi == 0 or self.allowNegativePOI)
            and self.indata.symmetric_tensor
            and self.indata.systematic_type == "normal"
            and ((not self.binByBinStat) or self.binByBinStatType == "normal")
        )

    def update_params(self):
        self.params.update(
            {
                "nobs": self.nobs,
                "lognobs": self.lognobs,
                "data_cov_inv": self.data_cov_inv,
                "theta0": self.theta0,
                "beta0": self.beta0,
                "beta": self.beta,
                "ubeta": self.ubeta,
                "logbeta0": self.logbeta0,
            }
        )

        leaves = jax.tree_util.tree_leaves(self.params)
        if all(isinstance(x, jax.Array) for x in leaves):
            logger.debug("All leaves are jax.Array")
        else:
            found = [x for x in leaves if isinstance(x, jax.Array)]
            logger.warning(f"Some leaves found that are not jax.Array: {found}")

    def init_blinding_values(self, unblind_parameter_expressions=[]):
        # Find parameters that match any regex
        compiled_expressions = [
            re.compile(expr) for expr in unblind_parameter_expressions
        ]

        unblind_parameters = [
            s
            for s in [
                *self.indata.signals,
                *[self.indata.systs[i] for i in self.indata.noigroupidxs],
            ]
            if any(regex.match(s.decode()) for regex in compiled_expressions)
        ]

        # check if dataset is an integer (i.e. if it is real data or not) and use this to choose the random seed
        is_dataobs_int = np.sum(
            np.equal(self.indata.data_obs, np.floor(self.indata.data_obs))
        )

        def deterministic_random_from_string(s, mean=0.0, std=5.0):
            # random value with seed taken based on string of parameter name
            if isinstance(s, str):
                s = s.encode("utf-8")

            if is_dataobs_int:
                s += b"_data"

            # Hash the string
            hash = hashlib.sha256(s).hexdigest()

            seed_seq = np.random.SeedSequence(int(hash, 16))
            rng = np.random.default_rng(seed_seq)

            value = rng.normal(loc=mean, scale=std)
            return value

        # multiply offset to nois
        self._blinding_values_theta = np.zeros(self.indata.nsyst, dtype=np.float64)
        for i in self.indata.noigroupidxs:
            param = self.indata.systs[i]
            if param in unblind_parameters:
                continue
            logger.debug(f"Blind parameter {param}")
            value = deterministic_random_from_string(param)
            self._blinding_values_theta[i] = value

        # add offset to pois
        self._blinding_values_poi = np.ones(self.npoi, dtype=np.float64)
        for i in range(self.npoi):
            param = self.indata.signals[i]
            if param in unblind_parameters:
                continue
            logger.debug(f"Blind signal strength modifier for {param}")
            value = deterministic_random_from_string(param)
            self._blinding_values_poi[i] = np.exp(value)

    def set_blinding_offsets(self, blind=True):
        if not self.do_blinding:
            return
        if blind:
            self._blinding_offsets_poi = jnp.array(self._blinding_values_poi)
            self._blinding_offsets_theta = jnp.array(self._blinding_values_theta)
        else:
            self._blinding_offsets_poi = jnp.ones(self.npoi, dtype=np.float64)
            self._blinding_offsets_theta = jnp.zeros(
                self.indata.nsyst, dtype=np.float64
            )

    def _default_beta0(self):
        if self.binByBinStatType == "gamma":
            return jnp.ones_like(self.indata.sumw)
        elif self.binByBinStatType == "normal":
            return jnp.zeros_like(self.indata.sumw)

    def prefit_covariance(self, unconstrained_err=0.0):
        # free parameters are taken to have zero uncertainty for the purposes of prefit uncertainties
        var_poi = jnp.zeros([self.npoi], dtype=self.indata.dtype)

        # nuisances have their uncertainty taken from the constraint term, but unconstrained nuisances
        # are set to a placeholder uncertainty (zero by default) for the purposes of prefit uncertainties
        var_theta = jnp.where(
            self.indata.constraintweights == 0.0,
            unconstrained_err**2,
            jnp.reciprocal(self.indata.constraintweights),
        )

        invhessianprefit = jnp.diag(jnp.concat([var_poi, var_theta], axis=0))
        return invhessianprefit

    @jax.jit
    def val_jac(self, fun, *args, **kwargs):
        with jnp.GradientTape() as t:
            val = fun(*args, **kwargs)
        jac = t.jacobian(val, self.x)

        return val, jac

    def set_nobs(self, values):
        if self.chisqFit and not self.externalCovariance:
            # covariance from data stat
            if jnp.any(values <= 0):
                raise RuntimeError(
                    "Bins in 'nobs <= 0' encountered, chi^2 fit can not be performed."
                )
        self.nobs = values
        # compute offset for poisson nll improved numerical precision in minimizatoin
        # the offset is chosen to give the saturated likelihood
        nobssafe = jnp.where(values == 0.0, 1.0, values)
        self.lognobs = jnp.log(nobssafe)

    def set_beta0(self, values):
        self.beta0 = values
        # compute offset for Gamma nll improved numerical precision in minimizatoin
        # the offset is chosen to give the saturated likelihood
        beta0safe = jnp.where(values == 0.0, 1.0, values)
        self.logbeta0 = jnp.log(beta0safe)

    def theta0defaultassign(self):
        self.theta0 = jnp.zeros([self.indata.nsyst], dtype=self.theta0.dtype)

    def xdefaultassign(self):
        if self.npoi == 0:
            self.x = self.theta0
        else:
            self.x = jnp.concat([self.xpoidefault, self.theta0], axis=0)

    def beta0defaultassign(self):
        self.set_beta0(self._default_beta0())

    def betadefaultassign(self):
        self.beta = self.beta0

    def defaultassign(self):
        self.cov = self.prefit_covariance(
            unconstrained_err=self.prefitUnconstrainedNuisanceUncertainty
        )
        self.theta0defaultassign()
        if self.binByBinStat:
            self.beta0defaultassign()
            self.betadefaultassign()
        self.xdefaultassign()
        if self.do_blinding:
            self.set_blinding_offsets(False)

    def bayesassign(self):
        # FIXME use theta0 as the mean and constraintweight to scale the width
        if self.npoi == 0:
            self.x = self.theta0 + jnp.random.normal(
                shape=self.theta0.shape, dtype=self.theta0.dtype
            )
        else:
            self.x = jnp.concat(
                [
                    self.xpoidefault,
                    self.theta0
                    + jnp.random.normal(
                        shape=self.theta0.shape, dtype=self.theta0.dtype
                    ),
                ],
                axis=0,
            )

        if self.binByBinStat:
            if self.binByBinStatType == "gamma":
                # FIXME this is only valid for beta0=beta=1 (but this should always be the case when throwing toys)
                betagen = (
                    jnp.random.gamma(
                        shape=[],
                        alpha=self.kstat * self.beta0 + 1.0,
                        beta=jnp.ones_like(self.kstat),
                        dtype=self.beta.dtype,
                    )
                    / self.kstat
                )

                betagen = jnp.where(self.kstat == 0.0, 0.0, betagen)
                self.beta = betagen
            elif self.binByBinStatType == "normal":
                self.beta = jnp.random.normal(
                    shape=[],
                    mean=self.beta0,
                    stddev=jnp.ones_like(self.beta0),
                    dtype=self.beta.dtype,
                )

    def frequentistassign(self):
        # FIXME use theta as the mean and constraintweight to scale the width
        self.theta0 = jnp.random.normal(
            shape=self.theta0.shape, dtype=self.theta0.dtype
        )
        if self.binByBinStat:
            if self.binByBinStatType == "gamma":
                # FIXME this is only valid for beta0=beta=1 (but this should always be the case when throwing toys)
                beta0gen = (
                    jnp.random.poisson(
                        shape=[],
                        lam=self.kstat * self.beta,
                        dtype=self.beta.dtype,
                    )
                    / self.kstat
                )

                beta0gen = jnp.where(
                    self.kstat == 0.0,
                    0.0,
                    beta0gen,
                )
                self.set_beta0(beta0gen)
            elif self.binByBinStatType == "normal":
                self.set_beta0(
                    jnp.random.normal(
                        shape=[],
                        mean=self.beta,
                        stddev=jnp.ones_like(self.beta0),
                        dtype=self.beta.dtype,
                    )
                )

    def toyassign(
        self,
        data_values=None,
        syst_randomize="frequentist",
        data_randomize="poisson",
        data_mode="expected",
        randomize_parameters=False,
    ):
        if syst_randomize == "bayesian":
            # randomize actual values
            self.bayesassign()
        elif syst_randomize == "frequentist":
            # randomize nuisance constraint minima
            self.frequentistassign()

        if data_mode == "expected":
            data_nom = self.expected_yield()
        elif data_mode == "observed":
            data_nom = data_values

        if data_randomize == "poisson":
            if self.externalCovariance:
                raise RuntimeError(
                    "Toys with external covariance only possible with data_randomize=normal"
                )
            else:
                self.set_nobs(
                    jnp.random.poisson(lam=data_nom, shape=[], dtype=self.nobs.dtype)
                )
        elif data_randomize == "normal":
            if self.externalCovariance:
                pdata = distrax.MultivariateNormalTriL(
                    loc=data_nom,
                    scale_tril=jnp.linalg.cholesky(jnp.linalg.inv(self.data_cov_inv)),
                )
                self.set_nobs(pdata.sample())
            else:
                self.set_nobs(
                    jnp.random.normal(
                        mean=data_nom,
                        stddev=jnp.sqrt(data_nom),
                        shape=[],
                        dtype=self.nobs.dtype,
                    )
                )
        elif data_randomize == "none":
            self.set_nobs(data_nom)

        # assign start values for nuisance parameters to constraint minima
        self.xdefaultassign()
        if self.binByBinStat:
            self.betadefaultassign()
        # set likelihood offset
        self.nexpnom = self.expected_yield()

        if randomize_parameters:
            # the special handling of the diagonal case here speeds things up, but is also required
            # in case the prefit covariance has zero for some uncertainties (which is the default
            # for unconstrained nuisances for example) since the multivariate normal distribution
            # requires a positive-definite covariance matrix
            if jh.is_diag(self.cov):
                self.x = jnp.random.normal(
                    shape=[],
                    mean=self.x,
                    stddev=jnp.sqrt(jnp.diagonal(self.cov)),
                    dtype=self.x.dtype,
                )
            else:
                pparms = distrax.MultivariateNormalTriL(
                    loc=self.x, scale_tril=jnp.linalg.cholesky(self.cov)
                )
                self.x = pparms.sample()
            if self.binByBinStat:
                self.beta = jnp.random.normal(
                    shape=[],
                    mean=self.beta0,
                    stddev=jnp.sqrt(self.indata.sumw2),
                    dtype=self.beta.dtype,
                )

    def _compute_impact_group(self, v, idxs):
        cov_reduced = jnp.take(self.cov[self.npoi :, self.npoi :], idxs, axis=0)
        cov_reduced = jnp.take(cov_reduced, idxs, axis=1)
        v_reduced = jnp.take(v, idxs, axis=1)
        invC_v = jnp.linalg.solve(cov_reduced, jnp.transpose(v_reduced))
        v_invC_v = jnp.einsum("ij,ji->i", v_reduced, invC_v)
        return jnp.sqrt(v_invC_v)

    @jax.jit
    def impacts_parms(self, hess):
        # impact for poi at index i in covariance matrix from nuisance with index j is C_ij/sqrt(C_jj) = <deltax deltatheta>/sqrt(<deltatheta^2>)
        cov_poi = self.cov[: self.npoi]
        cov_noi = jnp.take(self.cov[self.npoi :], self.indata.noigroupidxs)
        v = jnp.concat([cov_poi, cov_noi], axis=0)
        impacts = v / jnp.reshape(jnp.sqrt(jnp.diagonal(self.cov)), [1, -1])

        nstat = self.npoi + self.indata.nsystnoconstraint
        hess_stat = hess[:nstat, :nstat]
        inv_hess_stat = jnp.linalg.inv(hess_stat)

        if self.binByBinStat:
            # impact bin-by-bin stat
            val_no_bbb, grad_no_bbb, hess_no_bbb = self.loss_val_grad_hess(
                profile=False
            )

            hess_stat_no_bbb = hess_no_bbb[:nstat, :nstat]
            inv_hess_stat_no_bbb = jnp.linalg.inv(hess_stat_no_bbb)

            impacts_data_stat = jnp.sqrt(jnp.diagonal(inv_hess_stat_no_bbb))
            impacts_data_stat = jnp.reshape(impacts_data_stat, (-1, 1))

            impacts_bbb_sq = jnp.diagonal(inv_hess_stat - inv_hess_stat_no_bbb)
            impacts_bbb = jnp.sqrt(jnp.nn.relu(impacts_bbb_sq))  # max(0,x)
            impacts_bbb = jnp.reshape(impacts_bbb, (-1, 1))
            impacts_grouped = jnp.concat([impacts_data_stat, impacts_bbb], axis=1)
        else:
            impacts_data_stat = jnp.sqrt(jnp.diagonal(inv_hess_stat))
            impacts_data_stat = jnp.reshape(impacts_data_stat, (-1, 1))
            impacts_grouped = impacts_data_stat

        if len(self.indata.systgroupidxs):
            impacts_grouped_syst = jnp.map_fn(
                lambda idxs: self._compute_impact_group(v[:, self.npoi :], idxs),
                jnp.ragged.constant(self.indata.systgroupidxs, dtype=jnp.int32),
                fn_output_signature=jnp.TensorSpec(
                    shape=(impacts.shape[0],), dtype=jnp.float64
                ),
            )
            impacts_grouped_syst = jnp.transpose(impacts_grouped_syst)
            impacts_grouped = jnp.concat(
                [impacts_grouped_syst, impacts_grouped], axis=1
            )

        return impacts, impacts_grouped

    def _compute_global_impact_group(self, d_squared, idxs):
        gathered = jnp.take(d_squared, idxs, axis=-1)
        d_squared_summed = jnp.sum(gathered, axis=-1)
        return jnp.sqrt(d_squared_summed)

    @jax.jit
    def global_impacts_parms(self):
        # TODO migrate this to a physics model to avoid the below code which is largely duplicated

        idxs_poi = jnp.range(self.npoi, dtype=jnp.int64)
        idxs_noi = self.npoi + self.indata.noigroupidxs
        idxsout = jnp.concat([idxs_poi, idxs_noi], axis=0)

        dexpdx = jnp.one_hot(idxsout, depth=self.cov.shape[0], dtype=self.cov.dtype)

        cov_dexpdx = jnp.matmul(self.cov, dexpdx, transpose_b=True)

        var_total = jnp.diagonal(self.cov)
        var_total = jnp.take(var_total, idxsout)

        if self.binByBinStat:
            with jnp.GradientTape(persistent=True) as t2:
                t2.watch([self.x, self.ubeta])
                with jnp.GradientTape(persistent=True) as t1:
                    t1.watch([self.x, self.ubeta])
                    lc = self._compute_lc()
                    _1, _2, beta = self._compute_yields_with_beta(
                        profile=True, compute_norm=False, full=False
                    )
                    lbeta = self._compute_lbeta(beta)
                pdlbetadbeta = t1.gradient(lbeta, self.ubeta)
                dlcdx = t1.gradient(lc, self.x)
                dbetadx = t1.jacobian(beta, self.x)
            # pd2lbetadbeta2 is diagonal so we can use gradient instead of jacobian
            pd2lbetadbeta2_diag = t2.gradient(pdlbetadbeta, self.ubeta)
            # d2lcdx2 is diagonal so we can use gradient instead of jacobian
            d2lcdx2_diag = t2.gradient(dlcdx, self.x)
        else:
            with jnp.GradientTape() as t2:
                with jnp.GradientTape() as t1:
                    lc = self._compute_lc()
                dlcdx = t1.gradient(lc, self.x)
            # d2lcdx2 is diagonal so we can use gradient instead of jacobian
            d2lcdx2_diag = t2.gradient(dlcdx, self.x)

        # sc is the cholesky decomposition of d2lcdx2
        sc = jnp.diag(jnp.sqrt(d2lcdx2_diag), is_self_adjoint=True)

        impacts_x0 = sc @ cov_dexpdx
        impacts_theta0 = impacts_x0[self.npoi :]

        impacts_theta0 = jnp.transpose(impacts_theta0)
        impacts = impacts_theta0

        impacts_theta0_sq = jnp.square(impacts_theta0)
        var_theta0 = jnp.sum(impacts_theta0_sq, axis=-1)

        var_nobs = var_total - var_theta0

        if self.binByBinStat:
            # this the cholesky decomposition of pd2lbetadbeta2
            sbeta = jnp.diag(jnp.sqrt(pd2lbetadbeta2_diag), is_self_adjoint=True)

            impacts_beta0 = sbeta @ dbetadx @ cov_dexpdx

            var_beta0 = jnp.sum(jnp.square(impacts_beta0), axis=0)
            var_nobs -= var_beta0

            impacts_beta0 = jnp.sqrt(var_beta0)

        impacts_nobs = jnp.sqrt(var_nobs)

        if self.binByBinStat:
            impacts_grouped = jnp.stack([impacts_nobs, impacts_beta0], axis=-1)
        else:
            impacts_grouped = impacts_nobs[..., None]

        if len(self.indata.systgroupidxs):
            impacts_grouped_syst = jnp.map_fn(
                lambda idxs: self._compute_global_impact_group(impacts_theta0_sq, idxs),
                jnp.ragged.constant(self.indata.systgroupidxs, dtype=jnp.int64),
                fn_output_signature=jnp.TensorSpec(
                    shape=(impacts_theta0_sq.shape[0],), dtype=impacts_theta0_sq.dtype
                ),
            )
            impacts_grouped_syst = jnp.transpose(impacts_grouped_syst)
            impacts_grouped = jnp.concat(
                [impacts_grouped_syst, impacts_grouped], axis=1
            )

        return impacts, impacts_grouped

    def _pd2ldbeta2(self, profile=False):
        with jnp.GradientTape(watch_accessed_variables=False) as t2:
            t2.watch([self.ubeta])
            with jnp.GradientTape(watch_accessed_variables=False) as t1:
                t1.watch([self.ubeta])
                if profile:
                    val = self._compute_loss(profile=True)
                else:
                    # TODO this principle can probably be generalized to other parts of the code
                    # to further reduce special cases

                    # if not profiling, likelihood doesn't include the data contribution
                    _1, _2, beta = self._compute_yields_with_beta(
                        profile=False, compute_norm=False, full=False
                    )
                    lbeta = self._compute_lbeta(beta)
                    val = lbeta

            pdldbeta = t1.gradient(val, self.ubeta)
        if self.externalCovariance and profile:
            pd2ldbeta2_matrix = t2.jacobian(pdldbeta, self.ubeta)
            pd2ldbeta2 = jnp.linalg.LinearOperatorFullMatrix(
                pd2ldbeta2_matrix, is_self_adjoint=True
            )
        else:
            # pd2ldbeta2 is diagonal, so we can use gradient instead of jacobian
            pd2ldbeta2_diag = t2.gradient(pdldbeta, self.ubeta)
            pd2ldbeta2 = jnp.diag(pd2ldbeta2_diag, is_self_adjoint=True)
        return pd2ldbeta2

    def _dxdvars(self):
        with jnp.GradientTape() as t2:
            t2.watch([self.theta0, self.nobs, self.beta0])
            with jnp.GradientTape() as t1:
                t1.watch([self.theta0, self.nobs, self.beta0])
                val = self._compute_loss()
            grad = t1.gradient(val, self.x)
        pd2ldxdtheta0, pd2ldxdnobs, pd2ldxdbeta0 = t2.jacobian(
            grad, [self.theta0, self.nobs, self.beta0], unconnected_gradients="zero"
        )

        # cov is inverse hesse, thus cov ~ d2xd2l
        dxdtheta0 = -self.cov @ pd2ldxdtheta0
        dxdnobs = -self.cov @ pd2ldxdnobs
        dxdbeta0 = -self.cov @ pd2ldxdbeta0

        return dxdtheta0, dxdnobs, dxdbeta0

    def _expected_with_variance_optimized(self, fun_exp, skipBinByBinStat=False):
        # compute uncertainty on expectation propagating through uncertainty on fit parameters using full covariance matrix

        # FIXME this doesn't actually work for the positive semi-definite case
        invhesschol = jnp.linalg.cholesky(self.cov)

        # since the full covariance matrix with respect to the bin counts is given by J^T R^T R J, then summing RJ element-wise squared over the parameter axis gives the diagonal elements

        expected = fun_exp()

        # dummy vector for implicit transposition
        u = jnp.ones_like(expected)
        with jnp.GradientTape(watch_accessed_variables=False) as t1:
            t1.watch(u)
            with jnp.GradientTape() as t2:
                expected = fun_exp()
            # this returns dndx_j = sum_i u_i dn_i/dx_j
            Ju = t2.gradient(expected, self.x, output_gradients=u)
            Ju = jnp.transpose(Ju)
            Ju = jnp.reshape(Ju, [-1, 1])
            RJu = jnp.matmul(jax.lax.stop_gradient(invhesschol), Ju, transpose_a=True)
            RJu = jnp.reshape(RJu, [-1])
        RJ = t1.jacobian(RJu, u)
        sRJ2 = jnp.sum(RJ**2, axis=0)
        sRJ2 = jnp.reshape(sRJ2, jnp.shape(expected))
        if self.binByBinStat and not skipBinByBinStat:
            # add MC stat uncertainty on variance
            sumw2 = jnp.square(expected) / self.kstat
            sRJ2 = sRJ2 + sumw2
        return expected, sRJ2

    def _compute_expected(
        self, fun_exp, inclusive=True, profile=False, full=True, need_observables=True
    ):
        if need_observables:
            observables = self._compute_yields(
                inclusive=inclusive, profile=profile, full=full
            )
            expected = fun_exp(self.x, observables)
        else:
            expected = fun_exp(self.x)

        return expected

    def _expected_with_variance(
        self,
        fun_exp,
        compute_cov=False,
        compute_global_impacts=False,
        profile=False,
        inclusive=True,
        full=True,
        need_observables=True,
    ):
        # compute uncertainty on expectation propagating through uncertainty on fit parameters using full covariance matrix
        # FIXME switch back to optimized version at some point?

        def compute_derivatives(dvars):
            with jnp.GradientTape(watch_accessed_variables=False) as t:
                t.watch(dvars)
                expected = self._compute_expected(
                    fun_exp,
                    inclusive=inclusive,
                    profile=profile,
                    full=full,
                    need_observables=need_observables,
                )
                expected_flat = jnp.reshape(expected, (-1,))
            jacs = t.jacobian(
                expected_flat,
                dvars,
            )
            return expected, *jacs

        if self.binByBinStat:
            dvars = [self.x, self.ubeta]
            expected, dexpdx, pdexpdbeta = compute_derivatives(dvars)
        else:
            dvars = [self.x]
            expected, dexpdx = compute_derivatives(dvars)
            pdexpdbeta = None

        if compute_cov or (compute_global_impacts and self.binByBinStat):
            cov_dexpdx = jnp.matmul(self.cov, dexpdx, transpose_b=True)

        if compute_cov:
            expcov = dexpdx @ cov_dexpdx
        else:
            # matrix free calculation
            expvar_flat = jnp.einsum("ij,jk,ik->i", dexpdx, self.cov, dexpdx)
            expcov = None

        if pdexpdbeta is not None:
            pd2ldbeta2 = self._pd2ldbeta2(profile)
            pd2ldbeta2_pdexpdbeta = pd2ldbeta2.solve(pdexpdbeta, adjoint_arg=True)

            if compute_cov:
                expcov += pdexpdbeta @ pd2ldbeta2_pdexpdbeta
            else:
                expvar_flat += jnp.einsum("ik,ki->i", pdexpdbeta, pd2ldbeta2_pdexpdbeta)

        if compute_cov:
            expvar_flat = jnp.diagonal(expcov)

        expvar = jnp.reshape(expvar_flat, jnp.shape(expected))

        if compute_global_impacts:
            # the fully general contribution to the covariance matrix
            # for a factorized likelihood L = sum_i L_i can be written as
            # cov_i = dexpdx @ cov_x @ d2L_i/dx2 @ cov_x @ dexpdx.T
            # This is totally general and always adds up to the total covariance matrix

            # This can be factorized into impacts only if the individual contributions
            # are rank 1.  This is not the case in general for the data stat uncertainties,
            # in particular where postfit nexpected != nobserved and nexpected is not a linear
            # function of the poi's and nuisance parameters x

            # For the systematic and MC stat uncertainties this is equivalent to the
            # more conventional global impact calculation (and without needing to insert the uncertainty on
            # the global observables "by hand", which can be non-trivial beyond the Gaussian case)

            if self.binByBinStat:
                with jnp.GradientTape(persistent=True) as t2:
                    t2.watch([self.x, self.ubeta])
                    with jnp.GradientTape(persistent=True) as t1:
                        t1.watch([self.x, self.ubeta])
                        lc = self._compute_lc()
                        _1, _2, beta = self._compute_yields_with_beta(
                            profile=profile, compute_norm=False, full=False
                        )
                        lbeta = self._compute_lbeta(beta)
                    pdlbetadbeta = t1.gradient(lbeta, self.ubeta)
                    dlcdx = t1.gradient(lc, self.x)
                    dbetadx = t1.jacobian(beta, self.x)
                # pd2lbetadbeta2 is diagonal so we can use gradient instead of jacobian
                pd2lbetadbeta2_diag = t2.gradient(pdlbetadbeta, self.ubeta)
                # d2lcdx2 is diagonal so we can use gradient instead of jacobian
                d2lcdx2_diag = t2.gradient(dlcdx, self.x)
            else:
                with jnp.GradientTape() as t2:
                    with jnp.GradientTape() as t1:
                        lc = self._compute_lc()
                    dlcdx = t1.gradient(lc, self.x)
                # d2lcdx2 is diagonal so we can use gradient instead of jacobian
                d2lcdx2_diag = t2.gradient(dlcdx, self.x)

            # protect against inconsistency
            # FIXME this should be handled more generally e.g. through modification of
            # the constraintweights for prefit vs postfit, though special handling of the zero
            # uncertainty case would still be needed
            if (not profile) and self.prefitUnconstrainedNuisanceUncertainty != 0.0:
                raise NotImplementedError(
                    "Global impacts calculation not implemented for prefit case where prefitUnconstrainedNuisanceUncertainty != 0."
                )

            # sc is the cholesky decomposition of d2lcdx2
            sc = jnp.diag(jnp.sqrt(d2lcdx2_diag), is_self_adjoint=True)

            impacts_x0 = sc @ jnp.matmul(self.cov, dexpdx, transpose_b=True)
            impacts_theta0 = impacts_x0[self.npoi :]

            impacts_theta0 = jnp.transpose(impacts_theta0)
            impacts = impacts_theta0

            impacts_theta0_sq = jnp.square(impacts_theta0)
            var_theta0 = jnp.sum(impacts_theta0_sq, axis=-1)

            var_nobs = expvar_flat - var_theta0

            if self.binByBinStat:
                # this the cholesky decomposition of pd2lbetadbeta2
                sbeta = jnp.diag(jnp.sqrt(pd2lbetadbeta2_diag), is_self_adjoint=True)

                impacts_beta0 = jnp.zeros(
                    shape=(*self.beta.shape, *expvar_flat.shape), dtype=expvar.dtype
                )

                if pdexpdbeta is not None:
                    impacts_beta0 += sbeta @ pd2ldbeta2_pdexpdbeta

                if dbetadx is not None:
                    impacts_beta0 += sbeta @ dbetadx @ cov_dexpdx

                var_beta0 = jnp.sum(jnp.square(impacts_beta0), axis=0)
                var_nobs -= var_beta0

                impacts_beta0 = jnp.sqrt(var_beta0)

            impacts_nobs = jnp.sqrt(var_nobs)

            if self.binByBinStat:
                impacts_grouped = jnp.stack([impacts_nobs, impacts_beta0], axis=-1)
            else:
                impacts_grouped = impacts_nobs[..., None]

            if len(self.indata.systgroupidxs):
                impacts_grouped_syst = jnp.map_fn(
                    lambda idxs: self._compute_global_impact_group(
                        impacts_theta0_sq, idxs
                    ),
                    jnp.ragged.constant(self.indata.systgroupidxs, dtype=jnp.int64),
                    fn_output_signature=jnp.TensorSpec(
                        shape=(impacts_theta0_sq.shape[0],),
                        dtype=impacts_theta0_sq.dtype,
                    ),
                )
                impacts_grouped_syst = jnp.transpose(impacts_grouped_syst)

                impacts_grouped = jnp.concat(
                    [impacts_grouped_syst, impacts_grouped], axis=-1
                )

            impacts = jnp.reshape(impacts, [*expvar.shape, impacts.shape[-1]])
            impacts_grouped = jnp.reshape(
                impacts_grouped, [*expvar.shape, impacts_grouped.shape[-1]]
            )
        else:
            impacts = None
            impacts_grouped = None

        return expected, expvar, expcov, impacts, impacts_grouped

    def _expected_variations(
        self,
        fun_exp,
        correlations,
        inclusive=True,
        full=True,
        need_observables=True,
    ):
        with jnp.GradientTape() as t:
            # note that beta should only be profiled if correlations are taken into account
            expected = self._compute_expected(
                fun_exp,
                inclusive=inclusive,
                profile=correlations,
                full=full,
                need_observables=need_observables,
            )
            expected_flat = jnp.reshape(expected, (-1,))
        dexpdx = t.jacobian(expected_flat, self.x)

        if correlations:
            # construct the matrix such that the columns represent
            # the variations associated with profiling a given parameter
            # taking into account its correlations with the other parameters
            dx = self.cov / jnp.sqrt(jnp.diagonal(self.cov))[None, :]

            dexp = dexpdx @ dx
        else:
            dexp = dexpdx * jnp.sqrt(jnp.diagonal(self.cov))[None, :]

        new_shape = jnp.concat([jnp.shape(expected), [-1]], axis=0)
        dexp = jnp.reshape(dexp, new_shape)

        down = expected[..., None] - dexp
        up = expected[..., None] + dexp

        expvars = jnp.stack([down, up], axis=-1)

        return expvars

    # @jax.jit
    def _profile_beta(self, xval):
        nexp, norm, beta = _compute_yields_with_beta(
            self.params, xval, self.static_params
        )
        self.beta = beta

    def _compute_yields(self, xval, inclusive=True, profile=True, full=True):
        nexpcentral, normcentral, beta = _compute_yields_with_beta(
            self.params,
            xval,
            self.static_params,
            profile=profile,
            compute_norm=not inclusive,
            full=full,
        )
        if inclusive:
            return nexpcentral
        else:
            return normcentral

    @jax.jit
    def expected_with_variance(self, *args, **kwargs):
        return self._expected_with_variance(*args, **kwargs)

    @jax.jit
    def expected_variations(self, *args, **kwagrs):
        return self._expected_variations(*args, **kwagrs)

    def _residuals_profiled(
        self,
        fun,
    ):

        with jnp.GradientTape() as t:
            t.watch([self.theta0, self.nobs, self.beta0])
            expected = self._compute_expected(
                fun,
                inclusive=True,
                profile=True,
                full=False,
                need_observables=True,
            )
            observed = fun(None, self.nobs)
            residuals = expected - observed

            residuals_flat = jnp.reshape(residuals, (-1,))
        pdresdx, pdresdtheta0, pdresdnobs, pdresdbeta0 = t.jacobian(
            residuals_flat,
            [self.x, self.theta0, self.nobs, self.beta0],
            unconnected_gradients="zero",
        )

        # apply chain rule to take into account correlations with the fit parameters
        dxdtheta0, dxdnobs, dxdbeta0 = self._dxdvars()

        dresdtheta0 = pdresdtheta0 + pdresdx @ dxdtheta0
        dresdnobs = pdresdnobs + pdresdx @ dxdnobs
        dresdbeta0 = pdresdbeta0 + pdresdx @ dxdbeta0

        var_theta0 = jnp.where(
            self.indata.constraintweights == 0.0,
            jnp.zeros_like(self.indata.constraintweights),
            jnp.reciprocal(self.indata.constraintweights),
        )

        res_cov = dresdtheta0 @ (var_theta0[:, None] * jnp.transpose(dresdtheta0))

        if self.externalCovariance:
            res_cov_stat = dresdnobs @ jnp.linalg.solve(
                self.data_cov_inv, jnp.transpose(dresdnobs)
            )
        else:
            res_cov_stat = dresdnobs @ (self.nobs[:, None] * jnp.transpose(dresdnobs))

        res_cov += res_cov_stat

        if self.binByBinStat:
            pd2ldbeta2 = self._pd2ldbeta2(profile=False)
            pd2ldbeta2 = jnp.diagonal(pd2ldbeta2)

            with jnp.GradientTape() as t2:
                t2.watch([self.ubeta, self.beta0])
                with jnp.GradientTape() as t1:
                    t1.watch([self.ubeta, self.beta0])
                    _1, _2, beta = self._compute_yields_with_beta(
                        profile=False, compute_norm=False, full=False
                    )
                    lbeta = self._compute_lbeta(beta)

                dlbetadbeta = t1.gradient(lbeta, self.ubeta)
            pd2lbetadbetadbeta0 = t2.gradient(dlbetadbeta, self.beta0)

            var_beta0 = pd2ldbeta2 / pd2lbetadbetadbeta0**2

            if self.binByBinStatType == "gamma":
                var_beta0 = jnp.where(
                    self.betamask, jnp.zeros_like(var_beta0), var_beta0
                )

            res_cov_BBB = dresdbeta0 @ (var_beta0[:, None] * jnp.transpose(dresdbeta0))
            res_cov += res_cov_BBB

        return residuals, res_cov

    def _residuals(self, fun, fun_data):
        data, _0, data_cov = fun_data(self.nobs, self.data_cov_inv)
        pred, _0, pred_cov, _1, _2 = self._expected_with_variance(
            fun,
            profile=False,
            full=False,
            compute_cov=True,
            inclusive=True,
        )
        residuals = pred - data
        res_cov = pred_cov + data_cov
        return residuals, res_cov

    def _chi2(self, res, res_cov, ndf_reduction=0):
        res = jnp.reshape(res, (-1, 1))
        ndf = jnp.size(res) - ndf_reduction

        if ndf_reduction > 0:
            # covariance matrix is in general non invertible with ndf < n
            # compute chi2 using pseudo inverse
            chi_square_value = jnp.transpose(res) @ jnp.linalg.pinv(res_cov) @ res
        else:
            chi_square_value = jnp.transpose(res) @ jnp.linalg.solve(res_cov, res)

        return jnp.squeeze(chi_square_value), ndf

    @jax.jit
    def chi2(self, fun, fun_data=None, ndf_reduction=0, profile=False):
        if profile:
            residuals, res_cov = self._residuals_profiled(fun)
        else:
            residuals, res_cov = self._residuals(fun, fun_data)
        return self._chi2(residuals, res_cov, ndf_reduction)

    def expected_events(
        self,
        model,
        inclusive=True,
        compute_variance=True,
        compute_cov=False,
        compute_global_impacts=False,
        compute_variations=False,
        correlated_variations=False,
        profile=True,
        compute_chi2=False,
    ):

        if compute_variations and (
            compute_variance or compute_cov or compute_global_impacts
        ):
            raise NotImplementedError()

        fun = model.compute_flat if inclusive else model.compute_flat_per_process

        aux = [None] * 4
        if compute_cov or compute_variance or compute_global_impacts:
            exp, exp_var, exp_cov, exp_impacts, exp_impacts_grouped = (
                self.expected_with_variance(
                    fun,
                    profile=profile,
                    compute_cov=compute_cov,
                    compute_global_impacts=compute_global_impacts,
                    need_observables=model.need_observables,
                    inclusive=inclusive and not model.need_processes,
                )
            )
            aux = [exp_var, exp_cov, exp_impacts, exp_impacts_grouped]
        elif compute_variations:
            exp = self.expected_variations(
                fun,
                correlations=correlated_variations,
                inclusive=inclusive and not model.need_processes,
                need_observables=model.need_observables,
            )
        else:
            exp = self._compute_expected(
                fun,
                inclusive=inclusive and not model.need_processes,
                profile=profile,
                need_observables=model.need_observables,
            )

        if compute_chi2:
            chi2val, ndf = self.chi2(
                model.compute_flat,
                model._get_data,
                model.ndf_reduction,
                profile=profile,
            )

            aux.append(chi2val)
            aux.append(ndf)
        else:
            aux.append(None)
            aux.append(None)

        return exp, aux

    def expected_yield(self, profile=False, full=False):
        return self._compute_yields(self.x, inclusive=True, profile=profile, full=full)

    @jax.jit
    def _expected_yield_noBBB(self, xval, full=False):
        res, _ = self._compute_yields_noBBB(xval, full=full)
        return res

    def full_nll(self):
        return full_nll(self.params, self.x, self.static_params)

    def reduced_nll(self):
        return reduced_nll(self.params, self.x, self.static_params)

    def loss_val_grad_hess(self):
        return loss_val_grad_hess(self.params, self.x, self.static_params)

    def minimize(self):
        if self.is_linear:
            logger.info(
                "Likelihood is purely quadratic, solving by Cholesky decomposition instead of iterative fit"
            )

            # no need to do a minimization, simple matrix solve is sufficient
            val, grad, hess = self.loss_val_grad_hess()

            # use a Cholesky decomposition to easily detect the non-positive-definite case
            chol = jnp.linalg.cholesky(hess)

            # FIXME catch this exception to mark failed toys and continue
            if jnp.any(jnp.math.is_nan(chol)):
                raise ValueError(
                    "Cholesky decomposition failed, Hessian is not positive-definite"
                )

            del hess
            gradv = grad[..., None]
            dx = jnp.linalg.cholesky_solve(chol, -gradv)[:, 0]
            del chol

            self.x.assign_add(dx)
        else:
            start = time.time()

            def scipy_loss(xval):
                _0 = time.time()

                self.n_grad = self.n_grad + 1
                x = jnp.array(xval)

                _1 = time.time()
                self.time_hvp_copy_1 += _1 - _0

                val, grad = loss_val_grad(self.params, x, self.static_params)

                _2 = time.time()
                self.time_grad += _2 - _1

                val, grad = np.array(val), np.array(grad)

                self.time_grad_copy_2 += time.time() - _2

                # logger.debug(f"val = {val}")
                # logger.debug(f"grad = {grad}")
                return val, grad

            def scipy_hessp(xval, pval):
                _0 = time.time()

                self.n_hvp = self.n_hvp + 1
                x = jnp.array(xval)
                p = jnp.array(pval)

                _1 = time.time()
                self.time_hvp_copy_1 += _1 - _0

                hvp = loss_hessp(self.params, x, p, self.static_params)

                _2 = time.time()
                self.time_hvp += _2 - _1

                hvp = np.array(hvp)

                self.time_hvp_copy_2 += time.time() - _2

                # logger.debug(f"hvp = {hvp}")
                return hvp

            def scipy_hess(xval):
                _0 = time.time()
                self.n_hvp = self.n_hvp + 1
                x = jnp.array(xval)

                _1 = time.time()
                self.time_hvp_copy_1 += _1 - _0

                hess = loss_hess(self.params, x, self.static_params)

                _2 = time.time()
                self.time_hvp += _2 - _1

                if self.diagnostics:
                    raise NotImplementedError()
                    # # Compute condition number
                    # eigenvals = jnp.linalg.eigvals(hess)
                    # cond_number = jnp.max(eigenvals) / jnp.min(eigenvals)
                    # print(f"  - Condition number: {cond_number}")
                    # # Compute edmval
                    # edmval = 0.5 * jnp.dot(grad, linalg.solve(hess, grad))
                    # print(f"  - edmval: {edmval}")

                hess = np.array(hess)
                self.time_hvp_copy_2 += time.time() - _2

                return hess

            xval = self.x
            callback = FitterCallback(xval)

            if self.minimizer_method in [
                "trust-krylov",
            ]:
                info_minimize = dict(hessp=scipy_hessp)
            elif self.minimizer_method in [
                "trust-exact",
            ]:
                info_minimize = dict(hess=scipy_hess)
            else:
                info_minimize = dict()

            try:
                res = scipy.optimize.minimize(
                    scipy_loss,
                    xval,
                    method=self.minimizer_method,
                    jac=True,
                    tol=0.0,
                    callback=callback,
                    **info_minimize,
                )
            except Exception as ex:
                # minimizer could have called the loss or hessp functions with "random" values, so restore the
                # state from the end of the last iteration before the exception
                xval = callback.xval
                logger.debug(ex)
            else:
                xval = res["x"]
                logger.debug(res)
            self.x = xval

            self.time_minimizer = time.time() - start

        # force profiling of beta with final parameter values
        # TODO avoid the extra calculation and jitting if possible since the relevant calculation
        # usually would have been done during the minimization
        if self.binByBinStat:
            self._profile_beta(xval)

    def nll_scan(self, param, scan_range, scan_points, use_prefit=False):
        # make a likelihood scan for a single parameter
        # assuming the likelihood is minimized

        idx = np.where(self.parms.astype(str) == param)[0][0]

        # store current state of x temporarily
        xval = jnp.identity(self.x)

        param_offsets = np.linspace(0, scan_range, scan_points // 2 + 1)
        if not use_prefit:
            param_offsets *= self.cov[idx, idx] ** 0.5

        nscans = 2 * len(param_offsets) - 1
        dnlls = np.full(nscans, np.nan)
        scan_vals = np.zeros(nscans)

        # save delta nll w.r.t. global minimum
        nll_best = self.reduced_nll()
        # set central point
        dnlls[nscans // 2] = 0
        scan_vals[nscans // 2] = xval[idx]
        # scan positive side and negative side independently to profit from previous step
        for sign in [-1, 1]:
            param_scan_values = xval[idx] + sign * param_offsets
            for i, ixval in enumerate(param_scan_values):
                if i == 0:
                    continue

                self.x = jnp.tensor_scatter_nd_update(self.x, [[idx]], [ixval])

                def scipy_loss(xval):
                    self.x = xval
                    val, grad = loss_val_grad()
                    grad = grad
                    grad[idx] = 0  # Zero out gradient for the frozen parameter
                    return val, grad

                def scipy_hessp(xval, pval):
                    self.x = xval
                    pval[idx] = (
                        0  # Ensure the perturbation does not affect frozen parameter
                    )
                    p = jnp.convert_to_tensor(pval)
                    val, grad, hessp = loss_val_grad_hessp(p)
                    hessp = hessp
                    # TODO: worth testing modifying the loss/grad/hess functions to imply 1
                    # for the corresponding hessian element instead of 0,
                    # since this might allow the minimizer to converge more efficiently
                    hessp[idx] = (
                        0  # Zero out Hessian-vector product at the frozen index
                    )
                    return hessp

                res = scipy.optimize.minimize(
                    scipy_loss,
                    self.x,
                    method="trust-krylov",
                    jac=True,
                    hessp=scipy_hessp,
                )
                if res["success"]:
                    dnlls[nscans // 2 + sign * i] = self.reduced_nll() - nll_best
                    scan_vals[nscans // 2 + sign * i] = ixval

            # reset x to original state
            self.x = xval

        return scan_vals, dnlls

    def nll_scan2D(self, param_tuple, scan_range, scan_points, use_prefit=False):

        idx0 = np.where(self.parms.astype(str) == param_tuple[0])[0][0]
        idx1 = np.where(self.parms.astype(str) == param_tuple[1])[0][0]

        xval = jnp.identity(self.x)

        dsigs = np.linspace(-scan_range, scan_range, scan_points)
        if not use_prefit:
            x_scans = xval[idx0] + dsigs * self.cov[idx0, idx0] ** 0.5
            y_scans = xval[idx1] + dsigs * self.cov[idx1, idx1] ** 0.5
        else:
            x_scans = dsigs
            y_scans = dsigs

        best_fit = (scan_points + 1) // 2 - 1
        dnlls = np.full((len(x_scans), len(y_scans)), np.nan)
        nll_best = self.reduced_nll()
        dnlls[best_fit, best_fit] = 0
        # scan in a spiral around the best fit point
        dcol = -1
        drow = 0
        i = 0
        j = 0
        r = 1
        while r - 1 < best_fit:
            if i == r and drow == 1:
                drow = 0
                dcol = 1
            if j == r and dcol == 1:
                dcol = 0
                drow = -1
            elif i == -r and drow == -1:
                dcol = -1
                drow = 0
            elif j == -r and dcol == -1:
                drow = 1
                dcol = 0

            i += drow
            j += dcol

            if i == -r and j == -r:
                r += 1

            ix = best_fit - i
            iy = best_fit + j

            # print(f"i={i}, j={j}, r={r} drow={drow}, dcol={dcol} | ix={ix}, iy={iy}")

            self.x = jnp.tensor_scatter_nd_update(
                self.x, [[idx0], [idx1]], [x_scans[ix], y_scans[iy]]
            )

            def scipy_loss(xval):
                self.x = xval
                val, grad = self.loss_val_grad()
                grad = grad
                grad[idx0] = 0
                grad[idx1] = 0
                return val, grad

            def scipy_hessp(xval, pval):
                self.x = xval
                pval[idx0] = 0
                pval[idx1] = 0
                p = jnp.convert_to_tensor(pval)
                val, grad, hessp = self.loss_val_grad_hessp(p)
                hessp = hessp
                hessp[idx0] = 0
                hessp[idx1] = 0

                if np.allclose(hessp, 0, atol=1e-8):
                    return np.zeros_like(hessp)

                return hessp

            res = scipy.optimize.minimize(
                scipy_loss,
                self.x,
                method="trust-krylov",
                jac=True,
                hessp=scipy_hessp,
            )

            if res["success"]:
                dnlls[ix, iy] = self.reduced_nll() - nll_best

        self.x = xval
        return x_scans, y_scans, dnlls

    def contour_scan(self, param, nll_min, cl=1):

        def scipy_grad(xval):
            self.x = xval
            val, grad = self.loss_val_grad()
            return grad

        # def scipy_hessp(xval, pval):
        #     self.x = xval)
        #     p = jnp.convert_to_tensor(pval)
        #     val, grad, hessp = self.loss_val_grad_hessp(p)
        #     # print("scipy_hessp", val)
        #     return hessp

        def scipy_loss(xval):
            self.x = xval
            val = self.loss_val()
            return val - nll_min - 0.5 * cl**2

        nlc = scipy.optimize.NonlinearConstraint(
            fun=scipy_loss,
            lb=0,
            ub=0,
            jac=scipy_grad,
            hess=scipy.optimize.SR1(),  # TODO: use exact hessian or hessian vector product
        )

        # initial guess from covariance
        idx = np.where(self.parms.astype(str) == param)[0][0]
        xval = jnp.identity(self.x)

        xup = xval[idx] + self.cov[idx, idx] ** 0.5
        xdn = xval[idx] - self.cov[idx, idx] ** 0.5

        xval_init = xval

        intervals = np.full((2, len(self.parms)), np.nan)
        for i, sign in enumerate([-1.0, 1.0]):
            if sign == 1.0:
                xval_init[idx] = xdn
            else:
                xval_init[idx] = xup

            # Objective function and its derivatives
            def objective(params):
                return sign * params[idx]

            def objective_jac(params):
                jac = np.zeros_like(params)
                jac[idx] = sign
                return jac

            def objective_hessp(params, v):
                return np.zeros_like(v)

            res = scipy.optimize.minimize(
                objective,
                xval_init,
                method="trust-constr",
                jac=objective_jac,
                hessp=objective_hessp,
                constraints=[nlc],
                options={
                    "maxiter": 5000,
                    "xtol": 1e-10,
                    "gtol": 1e-10,
                    # "verbose": 3
                },
            )

            if res["success"]:
                intervals[i] = res["x"] - xval

            self.x = xval

        return intervals

    def contour_scan2D(self, param_tuple, nll_min, cl=1, n_points=16):
        # Not yet working
        def scipy_loss(xval):
            self.x = xval
            val, grad = self.loss_val_grad()
            return val

        def scipy_grad(xval):
            self.x = xval
            val, grad = self.loss_val_grad()
            return grad

        xval = jnp.identity(self.x)

        # Constraint function and its derivatives
        delta_nll = 0.5 * cl**2

        def constraint(params):
            return scipy_loss(params) - nll_min - delta_nll

        nlc = scipy.optimize.NonlinearConstraint(
            fun=constraint,
            lb=-np.inf,
            ub=0,
            jac=scipy_grad,
            hess=scipy.optimize.SR1(),
        )

        # initial guess from covariance
        xval_init = xval
        idx0 = np.where(self.parms.astype(str) == param_tuple[0])[0][0]
        idx1 = np.where(self.parms.astype(str) == param_tuple[1])[0][0]

        intervals = np.full((2, n_points), np.nan)
        for i, t in enumerate(np.linspace(0, 2 * np.pi, n_points, endpoint=False)):
            print(f"Now at {i} with angle={t}")

            # Objective function and its derivatives
            def objective(params):
                # coordinate center (best fit)
                x = params[idx0] - xval[idx0]
                y = params[idx1] - xval[idx1]
                return -(x**2 + y**2)

            def objective_jac(params):
                x = params[idx0] - xval[idx0]
                y = params[idx1] - xval[idx1]
                jac = np.zeros_like(params)
                jac[idx0] = -2 * x
                jac[idx1] = -2 * y
                return jac

            def objective_hessp(params, v):
                hessp = np.zeros_like(v)
                hessp[idx0] = -2 * v[idx0]
                hessp[idx1] = -2 * v[idx1]
                return hessp

            def constraint_angle(params):
                # coordinate center (best fit)
                x = params[idx0] - xval[idx0]
                y = params[idx1] - xval[idx1]
                return x * np.sin(t) - y * np.cos(t)

            def constraint_angle_jac(params):
                jac = np.zeros_like(params)
                jac[idx0] = np.sin(t)
                jac[idx1] = -np.cos(t)
                return jac

            # constraint on angle
            tc = scipy.optimize.NonlinearConstraint(
                fun=constraint_angle,
                lb=0,
                ub=0,
                jac=constraint_angle_jac,
                hess=scipy.optimize.SR1(),
            )

            res = scipy.optimize.minimize(
                objective,
                xval_init,
                method="trust-constr",
                jac=objective_jac,
                hessp=objective_hessp,
                constraints=[nlc, tc],
                options={
                    "maxiter": 10000,
                    "xtol": 1e-14,
                    "gtol": 1e-14,
                    # "verbose": 3
                },
            )

            print(res)

            if res["success"]:
                intervals[0, i] = res["x"][idx0]
                intervals[1, i] = res["x"][idx1]

            self.x = xval

        return intervals
