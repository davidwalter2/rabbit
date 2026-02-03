import numpy as np
import tensorflow as tf
from sme_constants import *
from sme_functions_precalc_mod import *

import h5py
from utilities.io_tools import input_tools
import pdb
from scripts.plotting.uncertainty_tools import *
from rabbit.poi_models.poi_model import POIModel

def tf_simpson(y, dx, axis=-1):
    y = tf.convert_to_tensor(y)

    # Move integration axis to the end
    y = tf.experimental.numpy.moveaxis(y, axis, -1)

    y0 = y[..., 0:-2:2]
    y1 = y[..., 1:-1:2]
    y2 = y[..., 2::2]

    return dx / 3.0 * tf.reduce_sum(y0 + 4.0 * y1 + y2, axis=-1)


class WilsonCoeffLeft(POIModel):
    def __init__(
        self, 
        indata,
        npoi, 
        poi_names,
        poi_defaults,
        left_tensor, #CL
        right_tensor, #CR
        **kwargs
    ):
        

        self.npoi = npoi
        self.pois = poi_names
        self.xpoidefault = poi_defaults
        self.is_linear = False
        self.allowNegativePOI = False
        
        
        
        times, pm, pn = get_hour_array()
        self.Q_vals = indata.axes
        self.Q_min = indata.axes()["pt_probe"][0]
        self.Q_max = indata.axes()["pt_probe"][1]
        all_q = []
        edges = indata.axes["pt_probe"]
        
        filename = f"{edges[0][0]}_to_{edges[-1][1]}_GeV_{len(edges)}_bins.pkl" 
        add_dir = "precomputed_sme/"
        all_precomputed = []
        
        for i in range(len(edges)):
            all_q.append(list(np.linspace(edges[i][0]**2, edges[i][1]**2, 101)))
            with open(add_dir + filename, "rb") as f:
                precomp_dict = pickle.load(f)
                all_precomputed.append(precomp_dict["values"][i][0]) ## this should only look at the up quark
        
        
        
        self.Q2_vals = all_q
        self.pm = pm
        self.pn = pn
        self.time = times
        self.precomputed_values = all_precomputed
        self.CR = right_tensor
        self.CL = left_tensor
        # self.sm_sigma = 
            
    @classmethod
    def compute(self, poi):
        
        ### poi[0] = left
        ## poi[1] = right
        total_cross_section = self.sm_sigma*tf.ones(len(self.Q2_vals))
        # modified_cross_section = np.array(len(self.Q2_vals))
        for i in range(len(self.Q2_vals)):
            integrand_values = [d_sigma_precomp(self.Q2_vals[i][j], self.pm, self.pn, poi[0]*self.CL, poi[1]*self.CR, precomputed_values = self.precomputed_values[i][j]) for j in range(len(self.Q2_vals[i]))]
            ### I'm not allowed to do simpson technically
            integral_liv = tf_simpson(integrand_values, self.Q2_vals[i])
            total_cross_section[i] += integral_liv 
        return total_cross_section/self.sm_sigma
            

        
        
        

if __name__ == "__main__":
    
    file_in = "/work/submit/jbenke/WRemnants/scripts/histmakers/"
    file_in_name = file_in + "mz_dilepton_liv_scetlib_dyturboCorr.hdf5"  # _maxFiles_20
    h5file = h5py.File(file_in_name, "r")
    results = input_tools.load_results_h5py(h5file)
    MC_Zmumu = results["ZmumuPostVFP"]["output"]
    data_output = results["dataPostVFP"]["output"]
    lumi_output = results["dataPostVFP"]["lumi_outout"]

    iso = MC_Zmumu["pos_iso"].get()
    trig = MC_Zmumu["pos_trig"].get()
    id_hist = MC_Zmumu["pos_ID"].get()
    global_hist = MC_Zmumu["pos_global"].get()


    weightsum = results["ZmumuPostVFP"]["weight_sum"]
    cross_sec = results["ZmumuPostVFP"]["dataset"]["xsec"]

    lumi_scaling = lumi_output["lumi_nom"].get()
    time_proj_low_all = data_output["time_proj"].get()

    time_proj_low = time_proj_low_all[{"mll": 9, "pt_tag": 3, "eta_tag": 3}]

    iso_corr = all_mc_corrections(
        iso.copy(), time_proj_low, lumi_scaling, weightsum, cross_sec
    )
    pdb.set_trace()

    g = WilsonCoeffLeft(iso_corr, 2, "cxxuL", "cxxuR", 1e-5, 1e-5, CL1, CR)
