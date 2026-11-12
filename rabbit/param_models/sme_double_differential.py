import numpy as np
import tensorflow as tf
from rabbit.param_models.liv_rapidity.basis import (
    LIV_QUARKS,
    YLL_FULL,
    load_or_compute_basis,
)
from rabbit.param_models.liv_rapidity.functions import (
    get_experiment,
    lab_coeffs,
    make_matrix,
)

from rabbit.param_models.param_model import ParamModel
from wums import logging

logger = logging.child_logger(__name__)

# SU(2) basis of arXiv:2011.02632 (Eqs. 1, 3, 15) and arXiv:2412.14305 (Tables 2, 4): per
# generation the independent coefficients are c_Q (left-handed doublet, shared by both
# quarks), c_U and c_D (right-handed singlets), tabulated as
#     c_u = (c_Q + c_U) / 2,  c_d = (c_Q + c_D) / 2,  d_u = (c_Q - c_U) / 2,
# with d_d = (c_Q - c_D) / 2 = c_u - c_d + d_u not independent. Inverting,
#     c_uL = c_u + d_u,  c_uR = c_u - d_u,  c_dL = c_u + d_u,  c_dR = 2 c_d - c_u - d_u,
# and the same for the second generation with u -> c, d -> s. A fit parameter (mode, quark),
# e.g. c..u, sets the chiral quark coefficients {(quark, chirality): weight} below with all
# other basis coefficients at zero (as for the single coefficient bounds of the papers).
# Third generation: as for the d=5 a_Sb, b_Sb of arXiv:2412.14305 ("For the third generation
# ... we consider a_Sb and b_Sb"), c_b and d_b are the plain c- and d-type coefficients of the
# bottom quark, c_b: c_bL = c_bR = 1, d_b: c_bL = -c_bR = 1, without an SU(2) relation (the
# top PDF is zero). Their Table 4 bounds are nearly equal (c_b^XY 4.1e-4, d_b^XY 4.3e-4), as
# g_L^2 + g_R^2 ~ g_L^2 - g_R^2 for down-type quarks.
SU2_BASIS = {
    ("c", "u"): {("u", "L"): 1.0, ("u", "R"): 1.0, ("d", "L"): 1.0, ("d", "R"): -1.0},
    ("c", "d"): {("d", "R"): 2.0},
    ("d", "u"): {("u", "L"): 1.0, ("u", "R"): -1.0, ("d", "L"): 1.0, ("d", "R"): -1.0},
    ("c", "c"): {("c", "L"): 1.0, ("c", "R"): 1.0, ("s", "L"): 1.0, ("s", "R"): -1.0},
    ("c", "s"): {("s", "R"): 2.0},
    ("d", "c"): {("c", "L"): 1.0, ("c", "R"): -1.0, ("s", "L"): 1.0, ("s", "R"): -1.0},
    ("c", "b"): {("b", "L"): 1.0, ("b", "R"): 1.0},
    ("d", "b"): {("b", "L"): 1.0, ("b", "R"): -1.0},
}
# d_s = (c_Q - c_S) / 2 = c_c - c_s + d_c is not independent. With the other coefficients
# at zero it is the same point as d_c alone (c_L = s_L = +1, c_R = s_R = -1), so d..s is
# accepted as an alias of d..c (identical template and bound). Likewise d_d = c_u - c_d + d_u
# alone is the same point as d_u alone. Aliases are not part of 'all', which would
# otherwise fit the same template twice.
SU2_ALIASES = {("d", "s"): ("d", "c"), ("d", "d"): ("d", "u")}
SU2_BASIS.update({alias: SU2_BASIS[target] for alias, target in SU2_ALIASES.items()})


class LIV_rapidity(ParamModel):
    # background processes scaled by the SME modulation in addition to the signals
    scaled_backgrounds = ["Ztautau"]
    # channel axes defining the SME templates (eta_probe is used as the rapidity)
    time_axis = "time"
    mll_axis = "mll"
    rapidity_axis = "eta_probe"
    # mll range of channels without an mll axis (efficiency channels at the Z pole)
    z_window = (76.0, 106.0)
    # rapidity range of channels without a rapidity axis
    rapidity_range = (-2.4, 2.4)
    # beam orientation (functions.EXPERIMENTS). None: the 'liv_experiment' entry of the
    # input file meta data if present, else CMS. Overridden by an 'experiment=ATLAS'
    # argument of --paramModel.
    experiment = None

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        experiment = cls.experiment
        yll_max = None
        mll_point = None
        rest = []
        for a in args:
            if a.lower().startswith("mllpoint="):
                mll_point = float(a.split("=", 1)[1])
            elif a.lower().startswith("experiment="):
                experiment = a.split("=", 1)[1]
            elif a.lower().startswith("yllmax="):
                yll_max = float(a.split("=", 1)[1])
            else:
                rest.append(a)
        args = rest
        metadata = getattr(indata, "metadata", {})
        if experiment is None:
            experiment = metadata.get("liv_experiment", "CMS")
        get_experiment(experiment)  # validate
        kwargs["experiment"] = experiment.upper()
        # |y_ll| range integrated for channels without a rapidity axis: a 'yllMax=X'
        # argument, else the 'liv_yll_max' entry of the meta data, else the class default
        if yll_max is None and "liv_yll_max" in metadata:
            yll_max = float(metadata["liv_yll_max"])
        if yll_max is not None:
            kwargs["rapidity_range"] = (-yll_max, yll_max)
        # channels without an mll axis evaluated at a single m_ll (dsigma/dm_ll at that
        # mass instead of integrated over z_window): an 'mllPoint=X' argument, else the
        # 'liv_mll_point' entry of the meta data
        if mll_point is None and metadata.get("liv_mll_point") is not None:
            mll_point = float(metadata["liv_mll_point"])
        if mll_point is not None:
            kwargs["mll_point"] = mll_point
        complete_args = []
        if "all" not in args:
            for coeff in args:
                if len(coeff) != 4:
                    if coeff[0] != "c":
                        if len(coeff) == 1:  ##u, for example
                            complete_args.append(f"cxx{coeff}")
                            complete_args.append(f"dxx{coeff}")
                            complete_args.append(f"cxy{coeff}")
                            complete_args.append(f"dxy{coeff}")
                            complete_args.append(f"cxz{coeff}")
                            complete_args.append(f"dxz{coeff}")
                            complete_args.append(f"cyz{coeff}")
                            complete_args.append(f"dyz{coeff}")
                            complete_args.append(f'ctx{coeff}')
                            complete_args.append(f'dtx{coeff}')
                            complete_args.append(f'cty{coeff}')
                            complete_args.append(f'dty{coeff}')
                        elif len(coeff) == 2:  # xx
                            complete_args.append(f"c{coeff}u")
                            complete_args.append(f"d{coeff}u")
                            complete_args.append(f"c{coeff}d")
                            complete_args.append(f'd{coeff}d')
                        elif len(coeff) == 3:  # xxu, for example
                            complete_args.append(f"c{coeff}")
                            complete_args.append(f"d{coeff}")
                    else:
                        if len(coeff) == 3:  # cxx
                            complete_args.append(f"{coeff}u")
                            complete_args.append(f"{coeff}d")
                            complete_args.append(f"{coeff}s")
                else:
                    complete_args.append(coeff)
        else:
            for mode, q in SU2_BASIS:
                if (mode, q) in SU2_ALIASES:
                    continue
                for direction in ["xx", "xy", "xz", "yz", "tx", "ty"]:
                    complete_args.append(f"{mode}{direction}{q}")

        # only coefficients of the SU(2) basis: explicitly requested others are an error,
        # those from the shorthand expansions (e.g. 'xx' -> dxxd) are dropped
        explicit = {a for a in args if len(a) == 4}
        invalid = [c for c in complete_args if (c[0], c[-1]) not in SU2_BASIS]
        if any(c in explicit for c in invalid):
            raise ValueError(
                f"Coefficients {[c for c in invalid if c in explicit]} are not in the SU(2) basis, "
                f"available: {[m + '..' + q for m, q in SU2_BASIS]}"
            )
        if invalid:
            logger.info(f"Dropping coefficients not in the SU(2) basis: {invalid}")
        complete_args = [c for c in complete_args if c not in invalid]

        return cls(indata, complete_args, **kwargs)

    def __init__(
        self,
        indata,
        coeff,
        scale=1e-6,
        experiment="CMS",
        rapidity_range=None,
        mll_point=None,
        **kwargs,
    ):
        self.indata = indata
        self.experiment = experiment
        self.mll_point = mll_point
        if mll_point is not None:
            logger.info(f"SME model: channels without an mll axis evaluate dsigma/dm_ll at m_ll = {mll_point:g} GeV")
        if rapidity_range is not None:
            self.rapidity_range = tuple(float(y) for y in rapidity_range)
            logger.info(f"SME model: channels without a rapidity axis integrate |y| < {self.rapidity_range[1]:g}")
        logger.info(f"SME model with the beam orientation of {experiment}")

        self.npoi = len(coeff)
        self.npou = 0

        self.params = np.array([c.encode() for c in coeff])

        self.xparamdefault = tf.zeros([self.npoi], dtype=self.indata.dtype)
        self.is_linear = True
        self.allowNegativeParam = True

        self.scale = scale

        coeffs = {
            # the xx parameter is c_XX - c_YY (c_XX = -c_YY = xx / 2), as tabulated in
            # arXiv:2011.02632 (Tables 2-5) and arXiv:2412.14305 (Table 4). c_XX + c_YY is
            # time independent and not a parameter of this model.
            'xx': 0.5 * (make_matrix(1, 1, 1.0) - make_matrix(2, 2, 1.0)),
            # signs as c^{XY}, c^{XZ}, c^{YZ} in Eq. 30 of arXiv:2011.02632 (verified against
            # lab_coeffs for ATLAS: identical c^{33}(T) for all four coefficients). ty and tx
            # are flipped in the same way (the paper does not give c^{03} in the SCF).
            'xy': make_matrix(1, 2, 1.0),
            'xz': make_matrix(1, 3, 1.0),
            'yz': make_matrix(2, 3, 1.0),
            'ty': make_matrix(0, 2, -1.0),
            'tx': make_matrix(0, 1, -1.0),
        }

        Q_ID = {'d': 1, 'u': 2, 's': 3, 'c': 4, 'b': 5}

        # the templates follow the unrolled fit input: for each unmasked channel the
        # bins are indexed by (time, mll, rapidity), with eta_probe used as the rapidity.
        # Channels without an mll axis (the efficiency channels at the Z pole) use
        # z_window, and the SM prediction of that (mll, eta) bin is applied to all
        # of their pt bins.
        bin_time, bin_int = [], []  # per fit bin: time bin (lo, hi) edges, index into int_bins
        int_bins = {}  # (Qlo, Qhi, ylo, yhi) -> index
        for name, info in self.indata.channel_info.items():
            if info.get("masked", False):
                continue
            axes = {a.name: a for a in info["axes"]}
            shape = [a.size for a in info["axes"]]
            if self.time_axis not in axes:
                raise ValueError(f"Channel {name} has no '{self.time_axis}' axis")

            def _edges(ax_name, default):
                if ax_name in axes:
                    e = axes[ax_name].edges
                    return list(zip(e[:-1], e[1:]))
                return [default]

            if self.mll_point is not None and self.mll_axis not in axes:
                # point bin (Qlo == Qhi): dsigma/dm_ll at m_ll = mll_point (basis.bin_xsec)
                mll_edges = [(self.mll_point, self.mll_point)]
            else:
                mll_edges = _edges(self.mll_axis, self.z_window)
            rap_edges = _edges(self.rapidity_axis, self.rapidity_range)

            # row-major bin indices of the channel, matching the unrolling
            idx = dict(zip([a.name for a in info["axes"]], np.indices(shape).reshape(len(shape), -1)))
            it = idx[self.time_axis]
            im = idx.get(self.mll_axis, np.zeros_like(it))
            ir = idx.get(self.rapidity_axis, np.zeros_like(it))

            # averaged over the time bin (local sidereal time in sidereal hours)
            t_edges = axes[self.time_axis].edges
            bin_time.append(np.stack([t_edges[it], t_edges[it + 1]], axis=-1))
            for m, r in zip(im, ir):
                key = (*map(float, mll_edges[m]), *map(float, rap_edges[r]))
                bin_int.append(int_bins.setdefault(key, len(int_bins)))
            logger.info(
                f"SME model channel {name}: {len(mll_edges)} mll x {len(rap_edges)} rapidity bins"
            )

        bin_time = np.concatenate(bin_time)
        bin_int = np.array(bin_int)
        int_bins = list(int_bins)

        # calculate integrals because of exponential rapidity drop. so cache the integrals
        sm_total, resp = load_or_compute_basis(int_bins, logger=logger)

        Q_IDX = {q: i for i, q in enumerate(LIV_QUARKS)}
        # resp[mode] is the response to c_qL = +-c_qR = 1 (mode c: +, d: -), so the
        # responses to the left and right handed coefficients alone are
        resp_chiral = {"L": 0.5 * (resp[0] + resp[1]), "R": 0.5 * (resp[0] - resp[1])}
        for c in coeff:
            if (c[0], c[-1]) not in SU2_BASIS:
                raise ValueError(f"Coefficient {c} is not in the SU(2) basis {list(SU2_BASIS)}")

        # lab frame coefficients [npoi, direction, time bin] averaged over each time bin.
        # They contain sidereal harmonics up to 2 omega, so a Gauss-Legendre average over
        # the bin is exact to machine precision (also for a few wide bins, e.g. 4 x 6 h).
        utimes, itime = np.unique(bin_time, axis=0, return_inverse=True)
        itime = itime.reshape(-1)

        gl_x, gl_w = np.polynomial.legendre.leggauss(16)
        gl_w = gl_w / gl_w.sum()
        values = np.zeros((self.npoi, len(bin_time)))
        for ip, c in enumerate(coeff):
            mat = coeffs[c[1:3]]
            cL = np.array(
                [
                    sum(
                        w * np.array([float(v) for v in lab_coeffs(t, mat, self.experiment)])
                        for t, w in zip(0.5 * (lo + hi) + 0.5 * (hi - lo) * gl_x, gl_w)
                    )
                    for lo, hi in utimes
                ]
            ).T  # [direction, time bin]
            # SU(2) basis coefficient -> chiral quark coefficients c_qL, c_qR (SU2_BASIS),
            # each with unit weight in the parton momentum shift (Eq. 15 of arXiv:2011.02632,
            # eta_fL,R = eta + c_fL,R, no factor 1/2). sme_model (liv/) instead uses
            # 1/2 (L +- R) per quark, i.e. a different, non SU(2) parametrisation.
            r = sum(
                w * resp_chiral[chirality][:, Q_IDX[Q_ID[q]]]
                for (q, chirality), w in SU2_BASIS[(c[0], c[-1])].items()
            )  # [direction, int bin]
            values[ip] = np.sum(cL[:, itime] * r[:, bin_int], axis=0)

        # change in cross section for a unit coefficient [npoi, nbins]
        self.sme = tf.constant(values, dtype=self.indata.dtype)
        # standard model cross section of the (mll, rapidity) bin of each fit bin [nbins]
        self.sm_sigma = tf.constant(sm_total[bin_int], dtype=self.indata.dtype)

        if self.sme.shape[1] != self.indata.nbins:
            raise ValueError(
                f"SME templates have {self.sme.shape[1]} bins but the fit has "
                f"{self.indata.nbins} (unmasked) bins"
            )

        # [1, nproc] mask: 1 for processes scaled by the SME modulation
        procs = [p.decode() if isinstance(p, bytes) else p for p in self.indata.procs]
        signals = [s.decode() if isinstance(s, bytes) else s for s in self.indata.signals]
        missing = [p for p in self.scaled_backgrounds if p not in procs]
        if missing:
            logger.warning(f"Processes {missing} to be scaled by the SME model not found in {procs}")
        scaled = [p in signals or p in self.scaled_backgrounds for p in procs]
        logger.info(f"SME model scales processes {[p for p, s in zip(procs, scaled) if s]}")
        self.proc_mask = tf.constant([scaled], dtype=self.indata.dtype)

    def compute(self, poi, full=False):
        shift = tf.tensordot(poi * self.scale, self.sme, axes=1) / self.sm_sigma  # [nbins]
        if full and self.indata.nbinsmasked:
            # masked channels are not modulated
            shift = tf.concat(
                [shift, tf.zeros([self.indata.nbinsmasked], dtype=self.indata.dtype)],
                axis=0,
            )
        # [nbins, nproc]: 1 + shift for scaled processes, 1 otherwise
        return 1.0 + shift[:, None] * self.proc_mask


class LIV_rapidity_pheno(LIV_rapidity):
    """Generator level projection, with (time, mll, yll) channels and no backgrounds"""

    scaled_backgrounds = []
    rapidity_axis = "yll"


class LIV_mll_pheno(LIV_rapidity_pheno):
    """Generator level projection, with (time, mll) channels integrated over all rapidities"""

    rapidity_range = (-YLL_FULL, YLL_FULL)


class LIV_rapidity_ATLAS(LIV_rapidity):
    experiment = "ATLAS"


class LIV_rapidity_pheno_ATLAS(LIV_rapidity_pheno):
    experiment = "ATLAS"


class LIV_mll_pheno_ATLAS(LIV_mll_pheno):
    experiment = "ATLAS"
