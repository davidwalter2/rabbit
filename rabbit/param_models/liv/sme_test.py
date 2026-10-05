import numpy as np
import tensorflow as tf
from rabbit.param_models.liv.sme_constants import *
from rabbit.param_models.liv.sme_functions import *

import pdb
from scripts.plotting.liv_variation_helpers import *
from rabbit.param_models.param_model import ParamModel
import tensorflow as tf

import matplotlib.pyplot as plt

# cxx_val = 1e-4*0.906
cxx_vals = [2.3e-4, 0.5e-4, 1e-3]


# for channel, info in indata.channel_info.items():
#     Q_vals = info["axes"][0]
#     Q_min = Q_vals[0][0]
#     Q_max = Q_vals[-1][1]

Q_min = 15
Q_max = 120
nTimeBins = 24
nEtaBins = 6
nPtBins = 14
Q_vals =  [15, 30, 40, 45, 50, 55, 60, 65, 70, 76, 106, 110, 115, 120]

coeff = "cxy"
quark = "u"
total_pt_eta_bins = nEtaBins*nPtBins*3 + nEtaBins*(nPtBins - 1)

mass = 6
add_dir = "/home/submit/jbenke/WRemnants/rabbit/rabbit/poi_models/precomputed_sigma/"
sme_L_filename = f"summation_{Q_min}_to_{Q_max}_GeV_{len(Q_vals)-1}_bins_{coeff}_{quark}_L.pkl"
sme_R_filename = f"summation_{Q_min}_to_{Q_max}_GeV_{len(Q_vals)-1}_bins_{coeff}_{quark}_R.pkl"
sm_filename = f"SM_{Q_min}_to_{Q_max}_GeV_{len(Q_vals)-1}_bins.pkl" 

#sme_left[time][mll]
with open(add_dir + sme_L_filename, "rb") as f:
    precomp_dict = pickle.load(f)
sme_left = tf.cast(np.array(precomp_dict["values"])[:, mass].flatten(), dtype = tf.float64)

# sme_left_full = tf.ones([total_pt_eta_bins, 1], dtype=tf.float64) * sme_left[None, :]
## NOT THRILLED THAT I HAVE THE SAME LINE MULTIPLE TIMES
sme_left_full = sme_left
sme_left_full = tf.reshape(sme_left_full, [-1, 1])[:, 0] 
            
            
#sme_right[time][mll]

with open(add_dir + sme_R_filename, "rb") as f:
    precomp_dict = pickle.load(f)
sme_right = tf.cast(np.array(precomp_dict["values"])[:, mass].flatten(), dtype = tf.float64)

# sme_right_full = sme_right[:, None] * tf.ones([1, total_pt_eta_bins], dtype=tf.float64) 
# sme_right_full = tf.ones([total_pt_eta_bins, 1], dtype=tf.float64) * sme_right[None, :]
sme_right_full = sme_right
sme_right_full = tf.reshape(sme_right_full, [-1, 1])[:, 0]

            
#sm_sigma[mll]
with open(add_dir + sm_filename, "rb") as f:
    precomp_dict = pickle.load(f)
# sm_sigma = tf.cast([precomp_dict["values"][9]]*(nTimeBins*total_pt_eta_bins), dtype = tf.float64) ## will need to expand this to duplicate along time axis

sm_sigma = tf.cast([precomp_dict["values"][mass]]*(nTimeBins), dtype = tf.float64) ## will need to expand this to duplicate along time axis

colors = ['r', 'b', 'g']
linestyles = ['solid', 'solid', 'dashed']
i = 0
for cxx_val in cxx_vals:
    flattened_xsec = (sm_sigma + 1/2*(sme_left_full+ sme_right_full)*cxx_val)/sm_sigma
    print(flattened_xsec)
    # times = np.linspace(0, nTimeBins*total_pt_eta_bins, nTimeBins*total_pt_eta_bins)
    times = np.linspace(0, nTimeBins, nTimeBins)

    plt.step(times, flattened_xsec, where = "post", label = f"$c^{{XY}}_u = {cxx_val}$", color = colors[i], linestyle = linestyles[i])
    i += 1
    plt.ylim([0.97, 1.03])
    plt.legend()
outdir = "/home/submit/jbenke/public_html/theory_predictions/"
date = "2026-05-22/"
filename = f"sigma_sme_sigma_sm_mass_bin_{mass}"
plt.savefig(outdir + date + filename, format = "png")