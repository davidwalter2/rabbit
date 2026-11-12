import pickle

import numpy as np
import tensorflow as tf
from rabbit.param_models.liv.sme_constants import *
from rabbit.param_models.liv.sme_functions import *

import h5py
from wremnants.utilities.io_tools import base_io
import pdb
from scripts.plotting.liv_variation_helpers import *
from rabbit.param_models.param_model import ParamModel
from wums import logging

logger = logging.child_logger(__name__)


class LIV(ParamModel):
    # background processes scaled by the SME modulation in addition to the signals
    scaled_backgrounds = ["Ztautau"]

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
                    if  coeff[0] != "c":# coeff[0] != "d" and 
                        if len(coeff) == 1: ##u, for example
                            complete_args.append(f"cxx{coeff}")   
                            complete_args.append(f"dxx{coeff}")  
                            complete_args.append(f"cxy{coeff}")   
                            complete_args.append(f"dxy{coeff}")  
                            complete_args.append(f"cxz{coeff}")   
                            complete_args.append(f"dxz{coeff}")  
                            complete_args.append(f"cyz{coeff}")   
                            complete_args.append(f"dyz{coeff}")
                        elif len(coeff) == 2: #xx
                            complete_args.append(f"c{coeff}u")   
                            complete_args.append(f"d{coeff}u")
                            complete_args.append(f"c{coeff}d")   
                            # complete_args.append(f"c{coeff}s")   
                            # complete_args.append(f"d{coeff}s")
                        elif len(coeff) == 3: #xxu, for example
                            complete_args.append(f"c{coeff}")   
                            complete_args.append(f"d{coeff}")  
                    else:
                        if len(coeff) == 3:  #cxx
                            complete_args.append(f"{coeff}u")   
                            complete_args.append(f"{coeff}d")
                            complete_args.append(f"{coeff}s")  

                else:
                    complete_args.append(coeff)
        
        else:
            quark = ['u', 'd', 's']
            all_coeffs = ["xx", "xy", "xz", "yz"]
            # for coeff in all_coeffs:
            #     complete_args.append(f"c{coeff}u")   
            #     complete_args.append(f"c{coeff}d")  
            #     complete_args.append(f"d{coeff}u") 
            #     # complete_args.append(f"d{coeff}d")   
            #     complete_args.append(f"c{coeff}s") 
            #     complete_args.append(f"d{coeff}s") 
            for q in quark:
                complete_args.append(f"cxx{q}")
                complete_args.append(f"cxy{q}")
                complete_args.append(f"cxz{q}")
                complete_args.append(f"cyz{q}")
                if q != 'd':
                    complete_args.append(f"dxx{q}")
                    complete_args.append(f"dxy{q}")
                    complete_args.append(f"dxz{q}")
                    complete_args.append(f"dyz{q}")
            
      
        return cls(indata, complete_args, **kwargs)
    

    def __init__(
        self, 
        indata,
        coeff,
        **kwargs
    ):
        ### i need to figure out how to put the parameters in the correct order. 

        z_pole_bin = 9
        self.indata = indata # since this line is in __init__ I don't think I need it

        self.npoi = len(coeff)
        self.npou = 0 # ?? i have no idea what the number of parameters of uninterest


        self.params = np.array([c.encode() for c in coeff])

        self.xparamdefault = tf.zeros([self.npoi], dtype=self.indata.dtype)
        self.is_linear = True
        self.allowNegativeParam = True

        def efficiency_flattening(input_hist):
            input_hist_full = input_hist[:, None]* tf.ones([1, self.nPtBins*self.nEtaBins], dtype=tf.float64) 
            input_hist_pt_clipped = input_hist[:, None]* tf.ones([1, (self.nPtBins-1)* self.nEtaBins], dtype=tf.float64) 
            input_hist_full = tf.reshape(input_hist_full, [-1, 1])[:, 0] #
            input_hist_pt_clipped = tf.reshape(input_hist_pt_clipped, [-1, 1])[:, 0] #
            full_hist = tf.concat([input_hist_full, input_hist_pt_clipped, input_hist_full, input_hist_full], axis = 0)
            full_hist = tf.reshape(full_hist, [-1, 1])[:, 0] 
            return full_hist
        
                   
        histmaker_dir = "/work/submit/jbenke/WRemnants/scripts/histmakers/"
        histmaker = (
            histmaker_dir + "mz_dilepton_liv_scetlib_dyturbo_CT18Z_N3p0LL_N2LO_Corr.hdf5"
        ) 

        h5file = h5py.File(histmaker, "r")
        results = base_io.load_results_h5py(h5file)
        data_output = results["SingleMuon_2016PostVFP"]["output"]
        ex_histogram = data_output["time_proj"].get()
                  
        self.Q_min = int(ex_histogram.axes["mll"][0][0])
        self.Q_max = int(ex_histogram.axes["mll"][-1][1])
        self.nTimeBins = len(ex_histogram.axes["time"])
        self.nMassBins = len(ex_histogram.axes["mll"])
        self.nPtBins = len(ex_histogram.axes["pt_probe"])
        self.nEtaBins = len(ex_histogram.axes["eta_probe"])

        
        liv_model = "/home/submit/jbenke/WRemnants/rabbit/rabbit/param_models/liv/precomputed_sigma/"
        sm_filename = f"SM_{self.Q_min}_to_{self.Q_max}_GeV_{self.nMassBins}_bins.pkl" 
    
        with open(liv_model + sm_filename, "rb") as f:
            precomp_dict = pickle.load(f)
        
        sm_sigma_eff = tf.cast([precomp_dict["values"][z_pole_bin]]*self.nTimeBins, dtype = tf.float64) 

        mass_axis = np.concatenate((precomp_dict["values"][:z_pole_bin], precomp_dict["values"][z_pole_bin+1:]))

        sm_sigma_mll_lower = tf.cast([precomp_dict["values"][:z_pole_bin]]*self.nTimeBins, dtype = tf.float64)
        sm_sigma_mll_lower = tf.reshape(sm_sigma_mll_lower, [-1, 1])[:, 0] 

        sm_sigma_mll_upper = tf.cast([precomp_dict["values"][z_pole_bin+1:]]*self.nTimeBins, dtype = tf.float64)
        sm_sigma_mll_upper = tf.reshape(sm_sigma_mll_upper, [-1, 1])[:, 0] 

        sm_sigma_mll = tf.concat([sm_sigma_mll_lower, sm_sigma_mll_upper], axis = 0)
        self.sm_sigma = tf.concat([sm_sigma_mll, efficiency_flattening(sm_sigma_eff)], axis = 0) ## flattens it

                
        sme_all = []
        print(coeff)
        for c in coeff:
            tensor = c[1:3]
            quark = c[-1]
            # tensor = c[:2] ### I don't remember what these are
            # quark = c[2]
            
            sme_L_filename = f"summation_{self.Q_min}_to_{self.Q_max}_GeV_{self.nMassBins}_bins_c{tensor}_{quark}_L.pkl"
            #sme[time][mll]
            with open(liv_model + sme_L_filename, "rb") as f:
                precomp_dict = pickle.load(f)

            sme_left_eff = tf.cast([precomp_dict["values"][:, z_pole_bin]], dtype = tf.float64)
            
            sme_left_eff = tf.reshape(sme_left_eff, [-1, 1])[:, 0] ## flattens it
            sme_left_eff_full = efficiency_flattening(sme_left_eff)
            
            sme_left_mass_axis = np.concatenate((precomp_dict["values"][:, :z_pole_bin], precomp_dict["values"][:, z_pole_bin+1:]), axis = 1)
            sme_left_mass_lower = tf.cast([precomp_dict["values"][:, :z_pole_bin]], dtype = tf.float64)
            sme_left_mass_lower = tf.reshape(sme_left_mass_lower, [-1, 1])[:, 0]

            sme_left_mass_upper = tf.cast([precomp_dict["values"][:, z_pole_bin+1:]], dtype = tf.float64)
            sme_left_mass_upper = tf.reshape(sme_left_mass_upper, [-1, 1])[:, 0]

            # sme_left_mll = tf.cast([sme_left_mass_axis], dtype = tf.float64)
            # sme_left_mll = tf.reshape(sme_left_mll, [-1, 1])[:, 0] ## flattens it

            sme_left_full = tf.concat([sme_left_mass_lower, sme_left_mass_upper, sme_left_eff_full], axis = 0)

            ### this is also not how they are taken apart. shit. 
            # sme_left_full = tf.concat([sme_left_eff_full, sme_left_mass_lower, sme_left_mass_upper], axis = 0)


            sme_R_filename = sme_L_filename[:-5] + "R" + sme_L_filename[-4:]
            #sme[time][mll]
            with open(liv_model + sme_R_filename, "rb") as f:
                precomp_dict = pickle.load(f)
            sme_right_eff = tf.cast([precomp_dict["values"][:, z_pole_bin]], dtype = tf.float64)
            sme_right_eff = tf.reshape(sme_right_eff, [-1, 1])[:, 0] ## flattens it
            sme_right_eff_full = efficiency_flattening(sme_right_eff)
            
            # sme_right_mass_axis = np.concatenate((precomp_dict["values"][:, :z_pole_bin], precomp_dict["values"][:, z_pole_bin+1:]), axis = 1)
            # sme_right_mll = tf.cast([sme_right_mass_axis], dtype = tf.float64)
            # sme_right_mll = tf.reshape(sme_right_mll, [-1, 1])[:, 0] ## flattens it

            sme_right_mass_lower = tf.cast([precomp_dict["values"][:, :z_pole_bin]], dtype = tf.float64)
            sme_right_mass_lower = tf.reshape(sme_right_mass_lower, [-1, 1])[:, 0]

            sme_right_mass_upper = tf.cast([precomp_dict["values"][:, z_pole_bin+1:]], dtype = tf.float64)
            sme_right_mass_upper = tf.reshape(sme_right_mass_upper, [-1, 1])[:, 0]


            ### h3_data_poi_low(time, mll), h3_data_poi_high(time, mll), h3_data_central_axes(time, pt_probe, eta_probe), h2_data_central_axes(time, pt_probe, eta_probe), h1_data_central_axes(time, pt_probe, eta_probe), h0_data_central_axes(time, pt_probe, eta_probe)


            sme_right_full = tf.concat([sme_right_mass_lower, sme_right_mass_upper, sme_right_eff_full], axis = 0)
            
            if c[0] == 'd':
                sme_all.append(1/2*(sme_left_full - sme_right_full))
            elif c[0] == 'c':
                sme_all.append(1/2*(sme_left_full + sme_right_full))
            # if c[-1] == "L":
            #     sme_all.append(sme_left_full)
            # elif c[-1] == "R":
            #     sme_all.append(sme_right_full)
        
        self.sme = tf.stack(sme_all, axis=0)  # [npoi, nbins]

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
        shift = tf.tensordot(poi * 1e-6, self.sme, axes=1) / self.sm_sigma  # [nbins]
        if full and self.indata.nbinsmasked:
            # masked channels are not modulated
            shift = tf.concat(
                [shift, tf.zeros([self.indata.nbinsmasked], dtype=self.indata.dtype)],
                axis=0,
            )
        # [nbins, nproc]: 1 + shift for scaled processes, 1 otherwise
        return 1.0 + shift[:, None] * self.proc_mask
