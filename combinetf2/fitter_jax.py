import jax
import jax.numpy as jnp
import numpy as np
import scipy
from wums import logging

from combinetf2.fitter import Fitter, FitterCallback
from combinetf2.jaxhelpers import edmval_cov

logger = logging.child_logger(__name__)


class FitterJax(Fitter):
    def __init__(self, bn, indata, options):
        super().__init__(bn, indata, options)

        # tf.config.experimental.enable_op_determinism()

        if options.eager:
            # TODO
            pass

        # tf.random.set_seed(options.seed)

        # # For debugging/reproducibility
        # jax.config.update('jax_debug_nans', True)
        # jax.config.update('jax_debug_infs', True)

    def assign_cov(self, values):
        self.cov = values

    @staticmethod
    def edmval_cov(*args, **kwargs):
        return edmval_cov(*args, **kwargs)

    def theta0defaultassign(self):
        self.theta0 = jnp.zeros([self.indata.nsyst], dtype=self.theta0.dtype)

    def xdefaultassign(self):
        if self.npoi == 0:
            self.x = self.theta0
        else:
            self.x = jnp.concat([self.xpoidefault, self.theta0], axis=0)

    def beta0defaultassign(self):
        self.beta0 = self._default_beta0()

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
        self.set_blinding_offsets(False)

    def expected_yield(self, profile=False, full=False):
        return self._compute_yields(inclusive=True, profile=profile, full=full)

    def _compute_yields(self, inclusive=True, profile=True, full=True):
        nexpcentral, normcentral, beta = self._compute_yields_with_beta(
            profile=profile,
            compute_norm=not inclusive,
            full=full,
        )
        if inclusive:
            return nexpcentral
        else:
            return normcentral

    def _compute_yields_noBBB(self, compute_norm=False, full=True):
        # compute_norm: compute yields for each process, otherwise inclusive
        # full: compute yields including masked channels
        poi = self.get_blinded_poi()
        theta = self.get_blinded_theta()

        rnorm = jnp.concatenate(
            [poi, jnp.ones(self.indata.nproc - poi.shape[0], dtype=self.indata.dtype)],
            axis=0,
        )

        mrnorm = jnp.expand_dims(rnorm, -1)
        ernorm = jnp.reshape(rnorm, [1, -1])

        normcentral = None
        if self.indata.symmetric_tensor:
            mthetaalpha = jnp.reshape(theta, [self.indata.nsyst, 1])
        else:
            # interpolation for asymmetric log-normal
            twox = 2.0 * theta
            twox2 = twox * twox
            alpha = 0.125 * twox * (twox2 * (3.0 * twox2 - 10.0) + 15.0)
            alpha = jnp.clip(alpha, -1.0, 1.0)

            thetaalpha = theta * alpha

            mthetaalpha = jnp.stack(
                [theta, thetaalpha], axis=0
            )  # now has shape [2,nsyst]
            mthetaalpha = jnp.reshape(mthetaalpha, [2 * self.indata.nsyst, 1])

        if self.indata.sparse:
            # For sparse operations, you'd need to implement sparse matrix multiplication
            # or convert to dense for JAX
            raise NotImplementedError(
                "Sparse operations not implemented in this JAX version"
            )
        else:
            if full or self.indata.nbinsmasked == 0:
                nbins = self.indata.nbinsfull
                logk = self.indata.logk
                norm = self.indata.norm
            else:
                nbins = self.indata.nbins
                logk = self.indata.logk[:nbins]
                norm = self.indata.norm[:nbins]

            if self.indata.symmetric_tensor:
                mlogk = jnp.reshape(
                    logk,
                    [nbins * self.indata.nproc, self.indata.nsyst],
                )
            else:
                mlogk = jnp.reshape(
                    logk,
                    [nbins * self.indata.nproc, 2 * self.indata.nsyst],
                )

            logsnorm = jnp.matmul(mlogk, mthetaalpha)
            logsnorm = jnp.reshape(logsnorm, [nbins, self.indata.nproc])

            if self.indata.systematic_type == "log_normal":
                snorm = jnp.exp(logsnorm)
                snormnorm = snorm * norm
                nexpcentral = jnp.matmul(snormnorm, mrnorm)
                nexpcentral = jnp.squeeze(nexpcentral, -1)
                if compute_norm:
                    normcentral = ernorm * snormnorm
            elif self.indata.systematic_type == "normal":
                normcentral = norm * ernorm + logsnorm
                nexpcentral = jnp.sum(normcentral, axis=-1)

        return nexpcentral, normcentral

    def _compute_yields_with_beta(self, profile=True, compute_norm=False, full=True):
        nexp, norm = self._compute_yields_noBBB(compute_norm, full=full)

        if self.binByBinStat:
            if profile:
                # analytic solution for profiled barlow-beeston lite parameters
                nexp_profile = nexp[: self.indata.nbins]
                beta0 = self.beta0[: self.indata.nbins]
                # Use stop_gradient equivalent (identity in forward pass)
                nobs0 = jax.lax.stop_gradient(self.nobs)

                if self.chisqFit:
                    if self.binByBinStatType == "gamma":
                        kstat = self.kstat[: self.indata.nbins]
                        betamask = self.betamask[: self.indata.nbins]

                        abeta = nexp_profile**2
                        bbeta = kstat * nobs0 - nexp_profile * self.nobs
                        cbeta = -kstat * nobs0 * beta0
                        beta = (
                            0.5
                            * (-bbeta + jnp.sqrt(bbeta**2 - 4.0 * abeta * cbeta))
                            / abeta
                        )
                        beta = jnp.where(betamask, beta0, beta)
                    elif self.binByBinStatType == "normal":
                        varbeta = self.indata.sumw2[: self.indata.nbins]
                        sbeta = jnp.sqrt(varbeta)
                        if self.externalCovariance:
                            raise NotImplementedError()
                            # sbeta_m = jnp.diag(sbeta)
                            # beta = linalg.lu_solve(
                            #     self.betaauxlu,
                            #     sbeta_m
                            #     @ self.data_cov_inv
                            #     @ ((self.nobs - nexp_profile)[:, None])
                            #     + beta0[:, None],
                            # )
                            # beta = jnp.squeeze(beta, axis=-1)
                        else:
                            beta = (
                                sbeta * (self.nobs - nexp_profile) + nobs0 * beta0
                            ) / (nobs0 + varbeta)
                else:
                    if self.binByBinStatType == "gamma":
                        kstat = self.kstat[: self.indata.nbins]
                        betamask = self.betamask[: self.indata.nbins]

                        beta = (self.nobs + kstat * beta0) / (nexp_profile + kstat)
                        beta = jnp.where(betamask, beta0, beta)
                    elif self.binByBinStatType == "normal":
                        varbeta = self.indata.sumw2[: self.indata.nbins]
                        sbeta = jnp.sqrt(varbeta)
                        abeta = sbeta
                        abeta = jnp.where(varbeta == 0.0, jnp.ones_like(abeta), abeta)
                        bbeta = varbeta + nexp_profile - sbeta * beta0
                        cbeta = (
                            sbeta * (nexp_profile - self.nobs) - nexp_profile * beta0
                        )
                        beta = (
                            0.5
                            * (-bbeta + jnp.sqrt(bbeta**2 - 4.0 * abeta * cbeta))
                            / abeta
                        )
                        beta = jnp.where(varbeta == 0.0, beta0, beta)

                if self.indata.nbinsmasked:
                    beta = jnp.concatenate(
                        [beta, self.beta0[self.indata.nbins :]], axis=0
                    )
            else:
                beta = self.beta

            # Add dummy tensor to allow convenient differentiation by beta even when profiling
            beta = beta + self.ubeta

            betasel = beta[: nexp.shape[0]]

            if self.binByBinStatType == "gamma":
                betamask = self.betamask[: nexp.shape[0]]
                nexp = jnp.where(betamask, nexp, nexp * betasel)
                if compute_norm:
                    norm = jnp.where(
                        betamask[..., None], norm, betasel[..., None] * norm
                    )
            elif self.binByBinStatType == "normal":
                varbeta = self.indata.sumw2[: nexp.shape[0]]
                sbeta = jnp.sqrt(varbeta)
                nexpnorm = nexp[..., None]
                nexp = nexp + sbeta * betasel
                if compute_norm:
                    # distribute the change in yields proportionally across processes
                    norm = (
                        norm + sbeta[..., None] * betasel[..., None] * norm / nexpnorm
                    )
        else:
            beta = None

        return nexp, norm, beta

    def _profile_beta(self):
        nexp, norm, beta = self._compute_yields_with_beta(full=False)
        self.beta = beta

    def full_nll(self):
        l, lfull = self._compute_nll()
        return lfull

    def reduced_nll(self):
        l, lfull = self._compute_nll()
        return l

    # @partial(jax.jit, static_argnums=(0,))
    def saturated_nll(self):
        nobs = self.nobs

        if self.chisqFit:
            lsaturated = jnp.array(0.0, dtype=self.nobs.dtype)
        else:
            nobsnull = jnp.equal(nobs, jnp.zeros_like(nobs))

            # saturated model
            nobssafe = jnp.where(nobsnull, jnp.ones_like(nobs), nobs)
            lognobs = jnp.log(nobssafe)

            lsaturated = jnp.sum(-nobs * lognobs + nobs, axis=-1)

        if self.binByBinStat:
            if self.binByBinStatType == "gamma":
                kstat = self.kstat
                beta0 = self.beta0
                lsaturated += jnp.sum(-kstat * beta0 * jnp.log(beta0) + kstat * beta0)
            elif self.binByBinStatType == "normal":
                # mc stat contribution to the saturated likelihood is zero in this case
                pass

        ndof = jnp.size(nobs) - self.npoi - self.indata.nsystnoconstraint

        return lsaturated, ndof

    def _compute_lc(self):
        # constraints
        theta = self.get_blinded_theta()
        lc = jnp.sum(
            self.indata.constraintweights * 0.5 * jnp.square(theta - self.theta0)
        )
        return lc

    def _compute_lbeta(self, beta):
        if self.binByBinStat:
            beta0 = self.beta0
            if self.binByBinStatType == "gamma":
                kstat = self.kstat

                lbetavfull = -kstat * beta0 * jnp.log(beta) + kstat * beta

                lbetav = -kstat * beta0 * jnp.log(beta) + kstat * (beta - 1.0)

                lbetafull = jnp.sum(lbetavfull)
                lbeta = jnp.sum(lbetav)
            elif self.binByBinStatType == "normal":
                lbetavfull = 0.5 * (beta - beta0) ** 2

                lbetafull = jnp.sum(lbetavfull)
                lbeta = lbetafull
        else:
            lbeta = None
            lbetafull = None
        return lbeta, lbetafull

    def _compute_nll_components(self, profile=True):
        nexpfullcentral, _, beta = self._compute_yields_with_beta(
            profile=profile,
            compute_norm=False,
            full=False,
        )

        nexp = nexpfullcentral

        if self.chisqFit:
            if self.externalCovariance:
                # Solve the system without inverting
                residual = jnp.reshape(self.nobs - nexp, [-1, 1])  # chi2 residual
                ln = lnfull = 0.5 * jnp.sum(
                    jnp.matmul(
                        residual.T,
                        jnp.matmul(self.data_cov_inv, residual),
                    )
                )
            else:
                # stop_gradient needed in denominator here because it should be considered
                # constant when evaluating global impacts from observed data
                ln = lnfull = 0.5 * jnp.sum(
                    (nexp - self.nobs) ** 2 / jax.lax.stop_gradient(self.nobs), axis=-1
                )
        else:
            nobsnull = jnp.equal(self.nobs, jnp.zeros_like(self.nobs))

            nexpsafe = jnp.where(nobsnull, jnp.ones_like(self.nobs), nexp)
            lognexp = jnp.log(nexpsafe)

            nexpnomsafe = jnp.where(nobsnull, jnp.ones_like(self.nobs), self.nexpnom)
            lognexpnom = jnp.log(nexpnomsafe)

            # final likelihood computation

            # poisson term
            lnfull = jnp.sum(-self.nobs * lognexp + nexp, axis=-1)

            # poisson term with offset to improve numerical precision
            ln = jnp.sum(
                -self.nobs * (lognexp - lognexpnom) + nexp - self.nexpnom, axis=-1
            )

        lc = lcfull = self._compute_lc()

        lbeta, lbetafull = self._compute_lbeta(beta)

        return ln, lc, lbeta, lnfull, lcfull, lbetafull, beta

    def _compute_nll(self, profile=True):
        ln, lc, lbeta, lnfull, lcfull, lbetafull, beta = self._compute_nll_components(
            profile=profile
        )
        l = ln + lc
        lfull = lnfull + lcfull

        if lbeta is not None:
            l = l + lbeta
            lfull = lfull + lbetafull

        return l, lfull

    def _compute_loss(self, profile=True):
        l, lfull = self._compute_nll(profile=profile)
        return l

    def loss_val(self, x):
        old_x = self.x
        self.x = x
        val = self._compute_loss()
        self.x = old_x
        return val

    def loss_val_grad(self):
        return jax.value_and_grad(self.loss_val)(self.x)

    def hessp(self, pval):
        # Compute Hessian-vector product using forward-mode AD
        def grad_fn(x):
            return jax.grad(self.loss_val)(x)

        _, hessp = jax.jvp(grad_fn, (self.x,), (pval,))

        return hessp

    def hess(self, profile=True):
        def loss_fn(x):
            old_x = self.x
            self.x = x
            val = self._compute_loss(profile=profile)
            self.x = old_x
            return val

        hess_fn = jax.hessian(loss_fn)

        hess = hess_fn(self.x)

        return hess

    def loss_val_grad_hess(self, profile=True):
        def loss_fn(x):
            old_x = self.x
            self.x = x
            val = self._compute_loss(profile=profile)
            self.x = old_x
            return val

        val = loss_fn(self.x)
        grad_fn = jax.grad(loss_fn)
        hess_fn = jax.hessian(loss_fn)

        grad = grad_fn(self.x)
        hess = hess_fn(self.x)

        return val, grad, hess

    def minimize(self):
        if self.is_linear:
            raise NotImplementedError()
            # print(
            #     "Likelihood is purely quadratic, solving by Cholesky decomposition instead of iterative fit"
            # )

            # # no need to do a minimization, simple matrix solve is sufficient
            # val, grad, hess = self.loss_val_grad_hess(self.x)

            # # use a Cholesky decomposition to easily detect the non-positive-definite case
            # try:
            #     chol = linalg.cholesky(hess)
            # except:
            #     raise ValueError(
            #         "Cholesky decomposition failed, Hessian is not positive-definite"
            #     )

            # gradv = grad[..., None]
            # dx = linalg.cho_solve((chol, True), -gradv)[:, 0]

            # self.x = self.x + dx
        else:

            def scipy_loss(xval):
                self.x = xval
                val, grad = self.loss_val_grad()
                logger.debug(f"val = {val}")
                logger.debug(f"grad = {grad}")
                return np.array(val), np.array(grad)

            def scipy_hessp(xval, pval):
                self.x = xval
                hessp = self.hessp(pval)
                logger.debug(f"hessp = {hessp}")
                return np.array(hessp)

            def scipy_hess(xval):
                self.x = xval
                hess = self.hess()
                # val, grad, hess = self.loss_val_grad_hess()
                if self.diagnostics:
                    raise NotImplementedError()
                    # # Compute condition number
                    # eigenvals = jnp.linalg.eigvals(hess)
                    # cond_number = jnp.max(eigenvals) / jnp.min(eigenvals)
                    # print(f"  - Condition number: {cond_number}")
                    # # Compute edmval
                    # edmval = 0.5 * jnp.dot(grad, linalg.solve(hess, grad))
                    # print(f"  - edmval: {edmval}")
                return np.array(hess)

            xval = np.array(self.x)
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

            self.x = jnp.array(xval)

        # force profiling of beta with final parameter values
        if self.binByBinStat:
            self._profile_beta()
