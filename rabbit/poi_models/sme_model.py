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

        #sm_sigma[mll]
        with open(add_dir + sm_filename, "rb") as f:
            precomp_dict = pickle.load(f)
        self.sm_sigma = np.array(precomp_dict["values"])

        self.Q2_vals = np.array([all_q])
        self.pm = pm ## will need to change this once I have multiple pm for
        self.pn = pn
        self.time = times
        self.CR = CR * 0 ## setting this to 0 for now. 
        self.CL = CL1 ## hard coding this for now
        self.nQbins = len(self.Q2_vals)
        self.nTimebins = 1
        self.xsec_mult = tf.Variable(tf.ones([self.nTimebins, self.nQbins], dtype = tf.float64))

            
    def compute(self, poi):
        
        ### poi[0] = left
        ## poi[1] = right

        
        # pdb.set_trace()
        #precomputed_values[mll][quark]
        #pm[time]
        #pn[time]
        #Q2_vals[time][mll][integration step]
        
        
        for k in range(self.nTimebins): #range(24)
            for i in range(self.nQbins):
                
                integrand_values = [d_sigma_precomp(self.pm[k], self.pn[k], poi[0]*self.CL, self.CR, precomputed_values = self.precomputed_values[i][j][0]) for j in range(len(self.Q2_vals[k][i]))]
                
                integral_liv = tf_simpson(integrand_values, self.Q2_vals[k][i])
                # pdb.set_trace()

                self.xsec_mult[k][i] = (integral_liv + self.sm_sigma[i])/self.sm_sigma[i]
        ## need to reshape the total_cross_section to be flat
        flattened_xsec = tf.reshape(self.xsec_mult, [self.nQbins * self.nTimebins])
        return flattened_xsec

                           

# if __name__ == "__main__":
    
#     file_in = "/work/submit/jbenke/WRemnants/scripts/histmakers/"
#     file_in_name = file_in + "mz_dilepton_liv_scetlib_dyturboCorr.hdf5"  # _maxFiles_20
#     h5file = h5py.File(file_in_name, "r")
#     results = input_tools.load_results_h5py(h5file)
#     MC_Zmumu = results["ZmumuPostVFP"]["output"]
#     data_output = results["dataPostVFP"]["output"]
#     lumi_output = results["dataPostVFP"]["lumi_outout"]

#     iso = MC_Zmumu["pos_iso"].get()
#     trig = MC_Zmumu["pos_trig"].get()
#     id_hist = MC_Zmumu["pos_ID"].get()
#     global_hist = MC_Zmumu["pos_global"].get()


#     weightsum = results["ZmumuPostVFP"]["weight_sum"]
#     cross_sec = results["ZmumuPostVFP"]["dataset"]["xsec"]

#     lumi_scaling = lumi_output["lumi_nom"].get()
#     time_proj_low_all = data_output["time_proj"].get()

#     time_proj_low = time_proj_low_all[{"mll": 9, "pt_tag": 3, "eta_tag": 3}]

#     iso_corr = all_mc_corrections(
#         iso.copy(), time_proj_low, lumi_scaling, weightsum, cross_sec
#     )
#     pdb.set_trace()

#     g = WilsonCoeffLeft(iso_corr, 2, "cxxuL", "cxxuR", 1e-5, 1e-5, CL1, CR)
