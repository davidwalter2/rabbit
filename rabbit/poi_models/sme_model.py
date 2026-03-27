import pickle

import h5py
import numpy as np
import tensorflow as tf
from wremnants.utilities.io_tools import base_io

from rabbit.poi_models.poi_model import POIModel


class LIV(POIModel):

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        """
        parsing the input arguments into the constructor, is has to be called as
        --poiModel Mixture coeff
        """
        ## coeff is of the form "xxuL"

        ### should write custom cases
        complete_args = []
        if "all" not in args:
            for coeff in args:
                if len(coeff) != 4:
                    if coeff[0] != "d" and coeff[0] != "c":
                        if len(coeff) == 1:  ##u, for example
                            complete_args.append(f"cxx{coeff}")
                            complete_args.append(f"dxx{coeff}")
                            complete_args.append(f"cxy{coeff}")
                            complete_args.append(f"dxy{coeff}")
                            complete_args.append(f"cxz{coeff}")
                            complete_args.append(f"dxz{coeff}")
                            complete_args.append(f"cyz{coeff}")
                            complete_args.append(f"dyz{coeff}")
                        elif len(coeff) == 2:  # xx
                            complete_args.append(f"c{coeff}u")
                            complete_args.append(f"d{coeff}u")
                            complete_args.append(f"c{coeff}d")
                            complete_args.append(f"c{coeff}s")
                            complete_args.append(f"d{coeff}s")
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
            all_coeffs = ["xx", "xy", "xz", "yz"]
            for coeff in all_coeffs:
                complete_args.append(f"c{coeff}u")
                complete_args.append(f"c{coeff}d")
                complete_args.append(f"d{coeff}u")
                complete_args.append(f"d{coeff}d")
                complete_args.append(f"c{coeff}s")
                complete_args.append(f"d{coeff}s")

        # i want to split this by generation so maybe do cxx1U so it goes type-generation-coefficient or dxx1U
        ## cxxu, cxxd, dxxu, dxxd

        ### expect coefficients of the form cu1 which is type-quark
        return cls(indata, complete_args, **kwargs)

    def __init__(self, indata, coeff, **kwargs):

        self.indata = indata
        self.is_linear = False

        self.allowNegativePOI = True
        self.npoi = len(coeff)
        self.pois = np.array([f"{coeff[i]}" for i in range(self.npoi)])
        self.xpoidefault = np.array([1] * self.npoi)

        def efficiency_flattening(input_hist):
            input_hist_full = input_hist[:, None] * tf.ones(
                [1, self.nEtaBins * self.nPtBins], dtype=tf.float64
            )
            input_hist_pt_clipped = input_hist[:, None] * tf.ones(
                [1, self.nEtaBins * (self.nPtBins - 1)], dtype=tf.float64
            )
            input_hist_full = tf.reshape(input_hist_full, [-1, 1])[:, 0]  #
            input_hist_pt_clipped = tf.reshape(input_hist_pt_clipped, [-1, 1])[:, 0]  #
            full_hist = tf.concat(
                [
                    input_hist_full,
                    input_hist_pt_clipped,
                    input_hist_full,
                    input_hist_full,
                ],
                axis=0,
            )
            full_hist = tf.reshape(full_hist, [-1, 1])[:, 0]
            return full_hist

        ref_file = "/work/submit/jbenke/WRemnants/scripts/histmakers/"
        ref_file_name = (
            ref_file + "mz_dilepton_liv_scetlib_dyturbo_CT18Z_N3p0LL_N2LO_Corr.hdf5"
        )
        h5file = h5py.File(ref_file_name, "r")
        results = base_io.load_results_h5py(h5file)
        data_output = results["SingleMuon_2016PostVFP"]["output"]
        ex_histogram = data_output["time_proj"].get()

        self.Q_min = int(ex_histogram.axes["mll"][0][0])
        self.Q_max = int(ex_histogram.axes["mll"][-1][1])
        self.nTimeBins = len(ex_histogram.axes["time"])
        self.nMassBins = len(ex_histogram.axes["mll"])
        self.nPtBins = len(ex_histogram.axes["pt_probe"])
        self.nEtaBins = len(ex_histogram.axes["eta_probe"])

        add_dir = (
            "/home/submit/jbenke/WRemnants/rabbit/rabbit/poi_models/precomputed_sigma/"
        )
        sm_filename = f"SM_{self.Q_min}_to_{self.Q_max}_GeV_{self.nMassBins}_bins.pkl"

        with open(add_dir + sm_filename, "rb") as f:
            precomp_dict = pickle.load(f)

        sm_sigma_eff = tf.cast(
            [precomp_dict["values"][9]] * self.nTimeBins, dtype=tf.float64
        )

        sm_sigma_mll = tf.cast(
            [precomp_dict["values"]] * self.nTimeBins, dtype=tf.float64
        )
        sm_sigma_mll = tf.reshape(sm_sigma_mll, [-1, 1])[:, 0]
        sm_sigma_mll_full = tf.concat(
            [sm_sigma_mll, sm_sigma_mll, sm_sigma_mll, sm_sigma_mll], axis=0
        )

        self.sm_sigma = tf.concat(
            [efficiency_flattening(sm_sigma_eff), sm_sigma_mll], axis=0
        )  ## flattens it

        sme_all = []
        print(coeff)
        for c in coeff:
            tensor = c[1:3]
            quark = c[-1]

            sme_L_filename = f"summation_{self.Q_min}_to_{self.Q_max}_GeV_{self.nMassBins}_bins_c{tensor}_{quark}_L.pkl"
            # sme[time][mll]
            with open(add_dir + sme_L_filename, "rb") as f:
                precomp_dict = pickle.load(f)
            sme_left_eff = tf.cast([precomp_dict["values"][:, 9]], dtype=tf.float64)
            sme_left_eff = tf.reshape(sme_left_eff, [-1, 1])[:, 0]  ## flattens it
            sme_left_eff_full = efficiency_flattening(sme_left_eff)

            sme_left_mll = tf.cast([precomp_dict["values"]], dtype=tf.float64)
            sme_left_mll = tf.reshape(sme_left_mll, [-1, 1])[:, 0]  ## flattens it

            sme_left_mll_full = tf.concat(
                [sme_left_mll, sme_left_mll, sme_left_mll, sme_left_mll], axis=0
            )

            sme_left_full = tf.concat([sme_left_eff_full, sme_left_mll], axis=0)

            sme_R_filename = sme_L_filename[:-5] + "R" + sme_L_filename[-4:]
            # sme[time][mll]
            with open(add_dir + sme_R_filename, "rb") as f:
                precomp_dict = pickle.load(f)
            sme_right_eff = tf.cast([precomp_dict["values"][:, 9]], dtype=tf.float64)
            sme_right_eff = tf.reshape(sme_right_eff, [-1, 1])[:, 0]  ## flattens it
            sme_right_eff_full = efficiency_flattening(sme_right_eff)

            sme_right_mll = tf.cast([precomp_dict["values"]], dtype=tf.float64)
            sme_right_mll = tf.reshape(sme_right_mll, [-1, 1])[:, 0]  ## flattens it

            sme_right_mll_full = tf.concat(
                [sme_right_mll, sme_right_mll, sme_right_mll, sme_right_mll], axis=0
            )
            sme_right_full = tf.concat([sme_right_eff_full, sme_right_mll], axis=0)

            if c[0] == "d":
                sme_all.append(1 / 2 * (sme_left_full - sme_right_full))
            elif c[0] == "c":
                sme_all.append(1 / 2 * (sme_left_full + sme_right_full))

        self.sme = np.array(sme_all)

    def compute(self, poi, full=False):
        flattened_xsec = self.sm_sigma / self.sm_sigma
        # for i in range(len(poi)):
        for i in range(poi.shape[0]):
            flattened_xsec += (self.sme[i] * poi[i] * 1e-6) / self.sm_sigma
        output = tf.reshape(flattened_xsec, [-1, 1])
        return output
