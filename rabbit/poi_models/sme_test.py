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

import tensorflow as tf


# cxx_val = 1e-4*0.906
cxx_val = 1e-5


# for channel, info in indata.channel_info.items():
#     Q_vals = info["axes"][0]
#     Q_min = Q_vals[0][0]
#     Q_max = Q_vals[-1][1]

Q_min = 15
Q_max = 120
nTimeBins = 24
Q_vals =  [15, 30, 40, 45, 50, 55, 60, 65, 70, 76, 106, 110, 115, 120]

coeff = "cxx"
quark = "u"

add_dir = "/home/submit/jbenke/WRemnants/rabbit/rabbit/poi_models/precomputed_sigma/"
sme_L_filename = f"summation_{Q_min}_to_{Q_max}_GeV_{len(Q_vals)-1}_bins_{coeff}_{quark}_L.pkl"
sme_R_filename = f"summation_{Q_min}_to_{Q_max}_GeV_{len(Q_vals)-1}_bins_{coeff}_{quark}_R.pkl"
sm_filename = f"SM_{Q_min}_to_{Q_max}_GeV_{len(Q_vals)-1}_bins.pkl" 

#sme_left[time][mll]
with open(add_dir + sme_L_filename, "rb") as f:
    precomp_dict = pickle.load(f)
sme_left = tf.cast(np.array(precomp_dict["values"])[:, 9].flatten(), dtype = tf.float64)
#sme_right[time][mll]
with open(add_dir + sme_R_filename, "rb") as f:
    precomp_dict = pickle.load(f)
sme_right = tf.cast(np.array(precomp_dict["values"])[:, 9].flatten(), dtype = tf.float64)
#sm_sigma[mll]
with open(add_dir + sm_filename, "rb") as f:
    precomp_dict = pickle.load(f)
sm_sigma = tf.cast([precomp_dict["values"][9]]*nTimeBins, dtype = tf.float64) ## will need to expand this to duplicate along time axis

        
flattened_xsec = (sm_sigma + sme_left*cxx_val + sme_right * 0)/sm_sigma
print(flattened_xsec)