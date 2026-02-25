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
    def __init__(
        self, 
        indata,
        # npoi, 
        # poi_names,
        # poi_defaults,
        # left_tensor, #CL
        # right_tensor, #CR
        **kwargs
    ):
        
        self.indata = indata
        self.is_linear = False

        self.allowNegativePOI = True
        self.npoi = 2
        self.pois = np.array(["cxx_u,L", "cxx_u,R"])
        self.xpoidefault = np.array([1, 1])


        # self.npoi = 1
        # self.pois = np.array(["cxx_u,L"])
        # self.xpoidefault = np.array([1])
                                     
        file_in = "/work/submit/jbenke/WRemnants/scripts/histmakers/"
        file_in_name = (
            file_in + "mz_dilepton_liv_scetlib_dyturbo_CT18Z_N3p0LL_N2LO_Corr.hdf5"
        )  # _maxFiles_20
        h5file = h5py.File(file_in_name, "r")
        results = input_tools.load_results_h5py(h5file)
        data_output = results["SingleMuon_2016PostVFP"]["output"]
        ex_histogram = data_output["time_proj"].get()
                  
        self.Q_min = int(ex_histogram.axes["mll"][0][0])
        self.Q_max = int(ex_histogram.axes["mll"][-1][1])
        self.nTimeBins = len(ex_histogram.axes["time"])
        self.nMassBins = len(ex_histogram.axes["mll"])

        self.coeff = "cxx"
        self.quark = "u"
        add_dir = "/home/submit/jbenke/WRemnants/rabbit/rabbit/poi_models/precomputed_sigma/"
        sme_L_filename = f"summation_{self.Q_min}_to_{self.Q_max}_GeV_{self.nMassBins}_bins_{self.coeff}_{self.quark}_L.pkl"
        sme_R_filename = f"summation_{self.Q_min}_to_{self.Q_max}_GeV_{self.nMassBins}_bins_{self.coeff}_{self.quark}_R.pkl"
        sm_filename = f"SM_{self.Q_min}_to_{self.Q_max}_GeV_{self.nMassBins}_bins.pkl" 
        
        ## for now select the largest mass bin
        #sme_left[time][mll]
        with open(add_dir + sme_L_filename, "rb") as f:
            precomp_dict = pickle.load(f)
        
        # self.sme_left = tf.cast(np.array(precomp_dict["values"][:, 9]).flatten(), dtype = tf.float64)
        # #sme_right[time][mll]
        # with open(add_dir + sme_R_filename, "rb") as f:
        #     precomp_dict = pickle.load(f)
            
        # self.sme_right = tf.cast(np.array(precomp_dict["values"][:, 9]).flatten(), dtype = tf.float64)
        # #sm_sigma[mll]
        # with open(add_dir + sm_filename, "rb") as f:
        #     precomp_dict = pickle.load(f)
            
        # self.sm_sigma = tf.cast([precomp_dict["values"][9]]*self.nTimeBins, dtype = tf.float64) ## will need to expand this to duplicate along time axis
        

        
        self.sme_left = tf.cast(np.array(precomp_dict["values"]).flatten(), dtype = tf.float64)
        #sme_right[time][mll]
        with open(add_dir + sme_R_filename, "rb") as f:
            precomp_dict = pickle.load(f)
            
        self.sme_right = tf.cast(np.array(precomp_dict["values"]).flatten(), dtype = tf.float64)
        #sm_sigma[mll]
        with open(add_dir + sm_filename, "rb") as f:
            precomp_dict = pickle.load(f)
            
        sm_sigma = tf.cast([precomp_dict["values"]]*self.nTimeBins, dtype = tf.float64) ## will need to expand this to duplicate along time axis
        
        
        self.sm_sigma = tf.reshape(sm_sigma, [-1, 1])[:, 0]
        
    def compute(self, poi):
        flattened_xsec = (self.sm_sigma + self.sme_left * poi[0]*1e-6+ self.sme_right*poi[1]*1e-6)/self.sm_sigma #   
        output = tf.reshape(flattened_xsec, [-1, 1])
        return output

