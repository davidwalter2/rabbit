import numpy as np
import tensorflow as tf
from rabbit.poi_models.sme_constants import *
from rabbit.poi_models.sme_functions import *

import h5py
from utilities.io_tools import input_tools
import pdb
from scripts.plotting.uncertainty_tools import *
from rabbit.poi_models.poi_model import POIModel
import tensorflow as tf


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
                    if coeff[0] != "d" and  coeff[0] != "c":
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
                            complete_args.append(f"c{coeff}s")   
                            complete_args.append(f"d{coeff}s")
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
    
    def __init__(
        self, 
        indata,
        coeff,
        **kwargs
    ):
        
        print(len(coeff))
        self.indata = indata
        self.is_linear = False

        self.allowNegativePOI = True
        self.npoi = len(coeff)
        self.pois = np.array([f"{coeff[i]}" for i in range(self.npoi)])
        self.xpoidefault = np.array([1]*self.npoi)

                   
        ref_file = "/work/submit/jbenke/WRemnants/scripts/histmakers/"
        ref_file_name = (
            ref_file + "mz_dilepton_liv_scetlib_dyturbo_all_bins.hdf5"
            # file_in + "mz_dilepton_liv_scetlib_dyturbo_CT18Z_N3p0LL_N2LO_Corr.hdf5"
        ) 
        h5file = h5py.File(ref_file_name, "r")
        results = input_tools.load_results_h5py(h5file)
        data_output = results["SingleMuon_2016PostVFP"]["output"]
        ex_histogram = data_output["time_proj"].get()
                  
        self.Q_min = int(ex_histogram.axes["mll"][0][0])
        self.Q_max = int(ex_histogram.axes["mll"][-1][1])
        self.nTimeBins = len(ex_histogram.axes["time"])
        self.nMassBins = len(ex_histogram.axes["mll"])
        
        add_dir = "/home/submit/jbenke/WRemnants/rabbit/rabbit/poi_models/precomputed_sigma/"
            
        sm_filename = f"SM_{self.Q_min}_to_{self.Q_max}_GeV_{self.nMassBins}_bins.pkl" 
        
        with open(add_dir + sm_filename, "rb") as f:
            precomp_dict = pickle.load(f)
            
        sm_sigma = tf.cast([precomp_dict["values"]]*self.nTimeBins, dtype = tf.float64)          
        self.sm_sigma = tf.reshape(sm_sigma, [-1, 1])[:, 0] ## flattens it
        
        
        sme_all = []
        print(coeff)
        for c in coeff:
            tensor = c[1:3]
            quark = c[-1]
            sme_L_filename = f"summation_{self.Q_min}_to_{self.Q_max}_GeV_{self.nMassBins}_bins_c{tensor}_{quark}_L.pkl"
            #sme[time][mll]
            with open(add_dir + sme_L_filename, "rb") as f:
                precomp_dict = pickle.load(f)
            sme_left = tf.cast(np.array(precomp_dict["values"]).flatten(), dtype = tf.float64)
            sme_R_filename = sme_L_filename[:-5] + "R" + sme_L_filename[-4:]
            #sme[time][mll]
            with open(add_dir + sme_R_filename, "rb") as f:
                precomp_dict = pickle.load(f)
            sme_right = tf.cast(np.array(precomp_dict["values"]).flatten(), dtype = tf.float64)
            
            if c[0] == 'd':
                sme_all.append(1/2*(sme_left - sme_right))
            elif c[0] == 'c':
                sme_all.append(1/2*(sme_left + sme_right))

        self.sme = np.array(sme_all)
        
        
    def compute(self, poi):
        flattened_xsec = self.sm_sigma/self.sm_sigma
        for i in range(len(poi)):
            flattened_xsec += (self.sme[i] * poi[i]*1e-6)/self.sm_sigma
        output = tf.reshape(flattened_xsec, [-1, 1])
        return output

