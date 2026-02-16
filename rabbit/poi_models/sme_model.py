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


def tf_simpson(y_list, x):

    y = tf.stack(y_list)  # shape: (N,)
    x = tf.convert_to_tensor(x, dtype=tf.float64)
       
    # Simpson's rule segments (assumes x can be non-uniform)
    h = x[2:] - x[:-2]  # distance between y[i] and y[i+2]
    simpson_sum = y[:-2] + 4*y[1:-1] + y[2:]
    integral = tf.reduce_sum(h / 6.0 * simpson_sum)
    
    # If even number of intervals, add trapezoid for last interval

    return integral


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
        self.xpoidefault = np.array([1e-5, 1e-5])


        # self.npoi = 1
        # self.pois = np.array(["cxx_u,L"])
        # self.xpoidefault = np.array([1e-5])
                                     
                                     
        
        # for channel, info in self.indata.channel_info.items():
        #     self.Q_vals = info["axes"][0]
        #     self.Q_min = self.Q_vals[0][0]
        #     self.Q_max = self.Q_vals[-1][1]
        
        self.Q_min = 15
        self.Q_max = 120
        self.nTimeBins = 24
        self.Q_vals =  [15, 30, 40, 45, 50, 55, 60, 65, 70, 76, 106, 110, 115, 120]

        self.coeff = "cxx"
        self.quark = "u"
        add_dir = "/home/submit/jbenke/WRemnants/rabbit/rabbit/poi_models/precomputed_sigma/"
        sme_L_filename = f"summation_{self.Q_min}_to_{self.Q_max}_GeV_{len(self.Q_vals)-1}_bins_{self.coeff}_{self.quark}_L.pkl"
        sme_R_filename = f"summation_{self.Q_min}_to_{self.Q_max}_GeV_{len(self.Q_vals)-1}_bins_{self.coeff}_{self.quark}_R.pkl"
        sm_filename = f"SM_{self.Q_min}_to_{self.Q_max}_GeV_{len(self.Q_vals)-1}_bins.pkl" 
        
        ## for now select the largest mass bin
        #sme_left[time][mll]
        with open(add_dir + sme_L_filename, "rb") as f:
            precomp_dict = pickle.load(f)
        self.sme_left = tf.cast(np.array(precomp_dict["values"])[:, 9].flatten(), dtype = tf.float64)
        #sme_right[time][mll]
        with open(add_dir + sme_R_filename, "rb") as f:
            precomp_dict = pickle.load(f)
        self.sme_right = tf.cast(np.array(precomp_dict["values"])[:, 9].flatten(), dtype = tf.float64)
        #sm_sigma[mll]
        with open(add_dir + sm_filename, "rb") as f:
            precomp_dict = pickle.load(f)
        self.sm_sigma = tf.cast([precomp_dict["values"][9]]*self.nTimeBins, dtype = tf.float64) ## will need to expand this to duplicate along time axis
        
    def compute(self, poi):
        flattened_xsec = (self.sm_sigma + self.sme_left*poi[0] + self.sme_right * poi[1])/self.sm_sigma
        output = tf.reshape(flattened_xsec, [-1, 1])
        return output

