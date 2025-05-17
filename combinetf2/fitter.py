import hashlib
import re

import numpy as np
from wums import logging

logger = logging.child_logger(__name__)


class Fitter:
    def __init__(self, bn, indata, options):
        self.bn = bn
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

        np.random.seed(options.seed)

        self.diagnostics = options.diagnostics
        self.minimizer_method = options.minimizerMethod

        self.chisqFit = options.chisqFit
        self.externalCovariance = options.externalCovariance
        self.prefitUnconstrainedNuisanceUncertainty = (
            options.prefitUnconstrainedNuisanceUncertainty
        )

        self.nsystgroupsfull = len(self.systgroupsfull)

        self.pois = []

        if 0 in options.toys:
            self.init_blinding_values(options.unblind)

        self.parms = np.concatenate([self.pois, self.indata.systs])

        self.allowNegativePOI = options.allowNegativePOI

        # observed number of events per bin
        self.data_cov_inv = None

        # determine if problem is linear (ie likelihood is purely quadratic)
        self.is_linear = (
            self.chisqFit
            and (self.npoi == 0 or self.allowNegativePOI)
            and self.indata.symmetric_tensor
            and self.indata.systematic_type == "normal"
            and ((not self.binByBinStat) or self.binByBinStatType == "normal")
        )

        if options.POIMode == "mu":
            self.npoi = self.indata.nsignals
            poidefault = options.POIDefault * bn.ones(
                [self.npoi], dtype=self.indata.dtype
            )
            for signal in self.indata.signals:
                self.pois.append(signal)
        elif options.POIMode == "none":
            self.npoi = 0
            poidefault = bn.zeros([], dtype=self.indata.dtype)
        else:
            raise Exception("unsupported POIMode")

        self._blinding_offsets_poi = bn.variable(
            bn.ones([self.npoi], dtype=self.indata.dtype),
            trainable=False,
            name="offset_poi",
        )
        self._blinding_offsets_theta = bn.variable(
            bn.zeros([self.indata.nsyst], dtype=self.indata.dtype),
            trainable=False,
            name="offset_theta",
        )

        self.parms = np.concatenate([self.pois, self.indata.systs])

        if self.allowNegativePOI:
            self.xpoidefault = poidefault
        else:
            self.xpoidefault = bn.sqrt(poidefault)

        # tf variable containing all fit parameters
        thetadefault = bn.zeros([self.indata.nsyst], dtype=self.indata.dtype)
        if self.npoi > 0:
            xdefault = bn.concat([self.xpoidefault, thetadefault], axis=0)
        else:
            xdefault = thetadefault

        self.x = bn.variable(xdefault, trainable=True, name="x")

        # observed number of events per bin
        self.nobs = bn.variable(self.indata.data_obs, trainable=False, name="nobs")

        if self.chisqFit:
            if self.externalCovariance:
                if self.indata.data_cov_inv is None:
                    raise RuntimeError("No external covariance found in input data.")
                # provided covariance
                self.data_cov_inv = self.indata.data_cov_inv
            else:
                # covariance from data stat
                if bn.reduce_any(self.nobs <= 0):
                    raise RuntimeError(
                        "Bins in 'nobs <= 0' encountered, chi^2 fit can not be performed."
                    )

        # constraint minima for nuisance parameters
        self.theta0 = bn.variable(
            bn.zeros([self.indata.nsyst], dtype=self.indata.dtype),
            trainable=False,
            name="theta0",
        )

        # FIXME for now this is needed even if binByBinStat is off because of how it is used in the global impacts
        #  and uncertainty band computations (gradient is allowed to be zero or None and then propagated or skipped only later)

        # global observables for mc stat uncertainty
        self.beta0 = bn.variable(self._default_beta0(), trainable=False, name="beta0")

        # nuisance parameters for mc stat uncertainty
        self.beta = bn.variable(self.beta0, trainable=False, name="beta")

        # dummy tensor to allow differentiation
        self.ubeta = bn.zeros_like(self.beta)

        if self.binByBinStat:
            if bn.reduce_any(self.indata.sumw2 < 0.0):
                raise ValueError("Negative variance for binByBinStat")

            if self.binByBinStatType == "gamma":
                self.kstat = self.indata.sumw**2 / self.indata.sumw2
                self.betamask = self.indata.sumw2 == 0.0
                self.kstat = bn.where(self.betamask, 1.0, self.kstat)
            elif self.binByBinStatType == "normal" and self.externalCovariance:
                # precompute decomposition of composite matrix to speed up
                # calculation of profiled beta values
                varbeta = self.indata.sumw2[: self.indata.nbins]
                sbeta = bn.sqrt(varbeta)
                sbeta_m = bn.diag_operator(sbeta)
                self.betaauxlu = bn.lu(
                    sbeta_m @ self.data_cov_inv @ sbeta_m
                    + bn.eye(self.data_cov_inv.shape[0], dtype=self.data_cov_inv.dtype)
                )

        self.nexpnom = bn.variable(
            self.expected_yield(), trainable=False, name="nexpnom"
        )

        # parameter covariance matrix
        self.cov = bn.variable(
            self.prefit_covariance(
                unconstrained_err=self.prefitUnconstrainedNuisanceUncertainty
            ),
            trainable=False,
            name="cov",
        )

    def init_blinding_values(self, unblind_parameter_expressions=[]):
        def compile_patterns(patterns):
            compiled = []
            for p in patterns:
                if p.startswith("r:"):
                    # Treat as regex, remove prefix
                    compiled.append(re.compile(p[2:]))
                else:
                    # Treat as exact string match
                    compiled.append(re.compile(rf"^{re.escape(p)}$"))
            return compiled

        # Find parameters that match any regex
        compiled_regexes = compile_patterns(unblind_parameter_expressions)
        unblind_parameters = [
            s
            for s in [*self.indata.procs, *self.indata.noigroups]
            if any(regex.search(s.decode()) for regex in compiled_regexes)
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
        if blind:
            self._blinding_offsets_poi.assign(self._blinding_values_poi)
            self._blinding_offsets_theta.assign(self._blinding_values_theta)
        else:
            self._blinding_offsets_poi.assign(np.ones(self.npoi, dtype=np.float64))
            self._blinding_offsets_theta.assign(
                np.zeros(self.indata.nsyst, dtype=np.float64)
            )

    def get_blinded_theta(self):
        theta = self.x[self.npoi :]
        theta = theta + self._blinding_offsets_theta
        return theta

    def get_params(self):
        return self.x, self.bn.diag_part(self.cov)

    def get_blinded_poi(self):
        xpoi = self.x[: self.npoi]
        if self.allowNegativePOI:
            poi = xpoi
        else:
            poi = self.bn.square(xpoi)
        poi = poi * self._blinding_offsets_poi
        return poi

    def _default_beta0(self):
        if self.binByBinStatType == "gamma":
            return self.bn.ones_like(self.indata.sumw)
        elif self.binByBinStatType == "normal":
            return self.bn.zeros_like(self.indata.sumw)

    def prefit_covariance(self, unconstrained_err=0.0):
        # free parameters are taken to have zero uncertainty for the purposes of prefit uncertainties
        var_poi = self.bn.zeros([self.npoi], dtype=self.indata.dtype)

        # nuisances have their uncertainty taken from the constraint term, but unconstrained nuisances
        # are set to a placeholder uncertainty (zero by default) for the purposes of prefit uncertainties
        var_theta = self.bn.where(
            self.indata.constraintweights == 0.0,
            unconstrained_err**2,
            self.bn.reciprocal(self.indata.constraintweights),
        )

        invhessianprefit = self.bn.diag(self.bn.concat([var_poi, var_theta], axis=0))
        return invhessianprefit
