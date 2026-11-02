import hashlib
import os

import numpy as np
import tensorflow as tf
from rabbit.param_models.liv_rapidity.functions import (
    integrate_xsec_yll,
    lab_coeffs,
    make_matrix,
    sidereal_seconds,
)

from rabbit.param_models.param_model import ParamModel
from wums import logging

logger = logging.child_logger(__name__)


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

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
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
            for q in ["u", "d", "s"]:
                complete_args.append(f"cxx{q}")
                complete_args.append(f"cxy{q}")
                complete_args.append(f"cxz{q}")
                complete_args.append(f"cyz{q}")
                complete_args.append(f'ctx{q}')
                complete_args.append(f'cty{q}')
                if q != 'd':
                    complete_args.append(f"dxx{q}")
                    complete_args.append(f"dxy{q}")
                    complete_args.append(f"dxz{q}")
                    complete_args.append(f"dyz{q}")
                    complete_args.append(f'dtx{q}')
                    complete_args.append(f'dty{q}')

        return cls(indata, complete_args, **kwargs)

    def __init__(
        self,
        indata,
        coeff,
        scale=1e-6,
        **kwargs,
    ):
        self.indata = indata

        self.npoi = len(coeff)
        self.npou = 0

        self.params = np.array([c.encode() for c in coeff])

        self.xparamdefault = tf.zeros([self.npoi], dtype=self.indata.dtype)
        self.is_linear = True
        self.allowNegativeParam = True

        self.scale = scale

        coeffs = {
            'xx': make_matrix(1, 1, 1.0) -  make_matrix(2, 2, 1.0),
            'xy': make_matrix(1, 2, -1.0),
            'xz': make_matrix(1, 3, -1.0),
            'yz': make_matrix(2, 3, -1.0),
            'ty': make_matrix(0, 2, 1.0),
            'tx': make_matrix(0, 1, 1.0),
        }

        Q_ID = {'d':1 , 'u':2, 's': 3}

        LIV_QUARKS = [1, 2, 3]
        EPS = 1e-6

        # the templates follow the unrolled fit input: for each unmasked channel the
        # bins are indexed by (time, mll, rapidity), with eta_probe used as the rapidity.
        # Channels without an mll axis (the efficiency channels at the Z pole) use
        # z_window, and the SM prediction of that (mll, eta) bin is applied to all
        # of their pt bins.
        bin_time, bin_int = [], []  # per fit bin: time center, index into int_bins
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

            mll_edges = _edges(self.mll_axis, self.z_window)
            rap_edges = _edges(self.rapidity_axis, self.rapidity_range)

            # row-major bin indices of the channel, matching the unrolling
            idx = dict(zip([a.name for a in info["axes"]], np.indices(shape).reshape(len(shape), -1)))
            it = idx[self.time_axis]
            im = idx.get(self.mll_axis, np.zeros_like(it))
            ir = idx.get(self.rapidity_axis, np.zeros_like(it))

            # evaluated at the lower bin edge, as in sme_model
            bin_time.append(axes[self.time_axis].edges[:-1][it])
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
        cache_key = hashlib.md5(
            repr((int_bins, LIV_QUARKS, EPS, "NNPDF31_nnlo_as_0118")).encode()
        ).hexdigest()[:12]
        cache_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "precomputed_sigma"
        )
        cache_file = os.path.join(cache_dir, f"liv_basis_{cache_key}.npz")

        if os.path.exists(cache_file):
            cached = np.load(cache_file)
            sm_total, resp = cached["sm_total"], cached["resp"]
        else:
            logger.info(
                f"LIV: integrating response basis for {len(int_bins)} (mll, rapidity) bins, will cache to {cache_file}"
            )
            sm_total = np.zeros(len(int_bins))
            # [mode (c/d), direction (c00/c33/c03), quark, (mll, rapidity) bin]
            resp = np.zeros((2, 3, len(LIV_QUARKS), len(int_bins)))
            for ib, (Qlo, Qhi, ylo, yhi) in enumerate(int_bins):
                for iq, q in enumerate(LIV_QUARKS):
                    s0 = integrate_xsec_yll(
                        [q], Qlo, Qhi, ylo, yhi, [0, 0, 0], [0, 0, 0]
                    )
                    # the cross section is a sum over quarks
                    sm_total[ib] += s0
                    for k in range(3):
                        e = [0.0, 0.0, 0.0]
                        e[k] = EPS
                        en = [-v for v in e]
                        resp[0, k, iq, ib] = (
                            integrate_xsec_yll([q], Qlo, Qhi, ylo, yhi, e, e) - s0
                        ) / EPS
                        resp[1, k, iq, ib] = (
                            integrate_xsec_yll([q], Qlo, Qhi, ylo, yhi, e, en) - s0
                        ) / EPS
                logger.info(f"LIV: bin {ib + 1}/{len(int_bins)} done")
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(cache_file, sm_total=sm_total, resp=resp)

        MODE = {"c": 0, "d": 1}
        Q_IDX = {q: i for i, q in enumerate(LIV_QUARKS)}

        # lab frame coefficients [npoi, direction, time bin] at the time bin lower edges
        utimes, itime = np.unique(bin_time, return_inverse=True)
        seconds = [sidereal_seconds(t) for t in utimes]
        values = np.zeros((self.npoi, len(bin_time)))
        for ip, c in enumerate(coeff):
            mat = coeffs[c[1:3]]
            cL = np.array(
                [[float(v) for v in lab_coeffs(t, mat)] for t in seconds]
            ).T  # [direction, time]
            # resp is the response to cL = +-cR = c, the factor 1/2 matches the
            # normalization of sme_model, 1/2 (L +- R), i.e. cL = +-cR = c / 2
            r = 0.5 * resp[MODE[c[0]], :, Q_IDX[Q_ID[c[-1]]]]  # [direction, int bin]
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
