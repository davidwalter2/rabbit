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
                    if len(coeff) == 2: ##xx, for example
                        ### u and d should be the same
                        complete_args.append(f"{coeff}uL")   
                        complete_args.append(f"{coeff}uR")  
                        complete_args.append(f"{coeff}dR") 
                        complete_args.append(f"{coeff}sL")   
                        complete_args.append(f"{coeff}sR") 
                    elif len(coeff) == 3: #xxu, for example
                        complete_args.append(f"{coeff}L")   
                        complete_args.append(f"{coeff}R")  
                else:
                    complete_args.append(coeff)
        
        else:
            all_coeffs = ["xx", "xy", "xz", "yz"]
            for coeff in all_coeffs:
                complete_args.append(f"{coeff}uL")   
                complete_args.append(f"{coeff}uR")  
                complete_args.append(f"{coeff}dR") 
                complete_args.append(f"{coeff}sL")   
                complete_args.append(f"{coeff}sR") 
            
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
        self.pois = np.array([f"c_{coeff[i]}" for i in range(self.npoi)])
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
        for c in coeff:
            tensor = c[:2]
            quark = c[2]
            hand = c[3]
            sme_filename = f"summation_{self.Q_min}_to_{self.Q_max}_GeV_{self.nMassBins}_bins_c{tensor}_{quark}_{hand}.pkl"

            #sme[time][mll]
            with open(add_dir + sme_filename, "rb") as f:
                precomp_dict = pickle.load(f)
            sme_all.append(tf.cast(np.array(precomp_dict["values"]).flatten(), dtype = tf.float64))
        self.sme = np.array(sme_all)
        
        
    def compute(self, poi):
        flattened_xsec = self.sm_sigma/self.sm_sigma
        for i in range(len(poi)):
            flattened_xsec += (self.sme[i] * poi[i]*1e-6)/self.sm_sigma
        output = tf.reshape(flattened_xsec, [-1, 1])
        return output

