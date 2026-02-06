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
        # self.npoi = npoi
        # self.pois = poi_names
        # self.xpoidefault = poi_defaults
        self.is_linear = False
        self.allowNegativePOI = False
        self.npoi = 1
        self.pois = np.array([s for s in self.indata.signals]) 
         
        self.xpoidefault = np.array([1e-5]) ### will need to eventually make this callable
        
        # for channel, info in self.indata.channel_info.items():
        #     self.Q_vals = info["axes"][0]
        #     self.Q_min = self.Q_vals[0][0]
        #     self.Q_max = self.Q_vals[-1][1]
        
        self.Q_min = 15
        self.Q_max = 120
        Q_vals_temp = [15, 30, 40, 45, 50, 55, 60, 65, 70, 76, 106, 110, 115, 120] ## need to not hardcode these
        
        self.Q_vals = [(Q_vals_temp[i], Q_vals_temp[i+1]) for i in range(len(Q_vals_temp) - 1)]
        times, pm, pn = get_hour_array()
        
        all_q = []
        
        # sme_filename = f"SME_{self.Q_min}_to_{self.Q_max}_GeV_{len(self.Q_vals)}_bins.pkl" 
        sm_filename = f"SM_{self.Q_min}_to_{self.Q_max}_GeV_{len(self.Q_vals)}_bins.pkl" 
        sme_filename = "SME_15_to_120_GeV_13_bins_time0.pkl"
        # sm_filename = "SM_15_to_120_GeV_13_bins.pkl"

        add_dir = "/home/submit/jbenke/WRemnants/rabbit/rabbit/poi_models/precomputed_sigma/"


        for i in range(len(self.Q_vals)):
            all_q.append(list(np.linspace(self.Q_vals[i][0]**2, self.Q_vals[i][1]**2, 101)))
            
        #precomputed_values[mll][nbins][quark]
        with open(add_dir + sme_filename, "rb") as f:
            precomp_dict = pickle.load(f)
            # all_precomputed_sme.append(precomp_dict["values"][i]) ## this should only look at the up quark
        
        self.precomputed_values = np.array(precomp_dict["values"])

        
        self.Q2_vals = np.array([all_q])
        self.pm = pm ## will need to change this once I have multiple pm for
        self.pn = pn
        self.time = times
        self.CR = CR * 0 ## setting this to 0 for now. 
        self.CL = CL1 ## hard coding this for now
        self.nQbins = self.Q2_vals.shape[1]
        self.nTimebins = 1
        self.xsec_mult = tf.Variable(tf.ones([self.nTimebins, self.nQbins], dtype = tf.float64))

        
        
        precomputed_Right = np.zeros([self.nTimebins, self.nQbins])
        precomputed_Left = np.zeros([self.nTimebins, self.nQbins])
       
        #precomputed_values[mll][quark]
        #pm[time]
        #pn[time]
        #Q2_vals[time][mll][integration step]
        
        for k in range(self.nTimebins): #range(24)
            for i in range(self.nQbins):
                pipi_L_int = GeV_to_pb*self.precomputed_values[i][:, 0, 0] * self.precomputed_values[i][:, 0, 2]
                pipj_L_int = GeV_to_pb*self.precomputed_values[i][:,0, 1] * self.precomputed_values[i][:,0, 2]
                
                pipi_R_int = GeV_to_pb*self.precomputed_values[i][:,0, 3] * self.precomputed_values[i][:,0, 5]
                pipj_R_int = GeV_to_pb*self.precomputed_values[i][:,0, 4] * self.precomputed_values[i][:,0, 5]
                
                integral_pipi_L = simpson(pipi_L_int, self.Q2_vals[k][i])
                integral_pipj_L = simpson(pipj_L_int, self.Q2_vals[k][i])
                integral_pipi_R = simpson(pipi_R_int, self.Q2_vals[k][i])
                integral_pipj_R = simpson(pipj_R_int, self.Q2_vals[k][i])
            

                contraction_p1p1_L = tf.einsum('mn,m,n->', self.CL, self.pm[k], self.pm[k])
                contraction_p1p2_L = tf.einsum('mn,m,n->', self.CL, self.pm[k], self.pn[k])
                contraction_p2p1_L = tf.einsum('mn,m,n->', self.CL, self.pn[k], self.pm[k])
                contraction_p2p2_L = tf.einsum('mn,m,n->', self.CL, self.pn[k], self.pn[k])
                
                contraction_pipi_L = (contraction_p1p1_L + contraction_p2p2_L)
                contraction_pipj_L = (contraction_p1p2_L + contraction_p2p1_L)
                        
                contraction_p1p1_R = tf.einsum('mn,m,n->', 0*self.CR, self.pm[k], self.pm[k])
                contraction_p1p2_R = tf.einsum('mn,m,n->', 0*self.CR, self.pm[k], self.pn[k])
                contraction_p2p1_R = tf.einsum('mn,m,n->', 0*self.CR, self.pn[k], self.pm[k])
                contraction_p2p2_R = tf.einsum('mn,m,n->', 0*self.CR, self.pn[k], self.pn[k])

                contraction_pipi_R = (contraction_p1p1_R + contraction_p2p2_R)
                contraction_pipj_R = (contraction_p1p2_R + contraction_p2p1_R)
        
        
        
                precomputed_Left[k][i] = integral_pipi_L * contraction_pipi_L + contraction_pipj_L * integral_pipj_L
                precomputed_Right[k][i] = integral_pipi_R * contraction_pipi_R + integral_pipj_R * contraction_pipj_R

        
        #sm_sigma[mll]
        with open(add_dir + sm_filename, "rb") as f:
            precomp_dict = pickle.load(f)
        self.sm_sigma = tf.cast(precomp_dict["values"]*self.nTimebins, dtype = tf.float64) ## will need to expand this to duplicate along time axis


        self.sme_left = tf.cast(precomputed_Left.flatten(), dtype = tf.float64)
        self.sme_right = tf.cast(precomputed_Right.flatten(), dtype = tf.float64)
        
            
    def compute(self, poi):
        
        flattened_xsec = (self.sm_sigma + self.sme_left*poi[0] + self.sme_right * 0)/self.sm_sigma
        
        rnorm = tf.ones(self.indata.nproc, dtype=self.indata.dtype)
        rnorm = tf.reshape(rnorm, [1, -1])
        
        return flattened_xsec*rnorm

    