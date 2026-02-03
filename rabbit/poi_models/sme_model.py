import numpy as np
import tensorflow as tf
from sme_constants import *
from sme_functions_precal_mod import *
from rabbit.poi_models.poi_model import POIModel

class WilsonCoeffLeft(POIModel):
    
    def __init__(
        self, 
        indata,
        npoi, 
        poi_names,
        poi_defaults,
        e_f, 
        flavor, 
        g_fR,
        g_fL,
        left_tensor, #CL
        right_tensor, #CR
    ):
        self.npoi = npoi
        self.pois = poi_names
        self.xpoidefault = poi_defaults
        self.is_linear = False
        self.allowNegativePOI = False
            
        

        self.CL = left_tensor
        self.CR = right_tensor
       
        
        times, pm, pn = get_hour_array()
        self.Q_vals = indata.axes
        num_steps_Q2 = 100
        self.Q_min = indata.axes()["pt_probe"][0]
        self.Q_max = indata.axes()["pt_probe"][1]
        all_q = []
        for i in range(len(indata.axes()["pt_probe"]) - 2):
            all_q.append(list(np.linspace(indata.axes()["pt_probe"][i], indata.axes()["pt_probe"][i+1], num_steps_Q2)))
            

        

    @classmethod
    def compute(self, poi):
        
        
        integral1 = integrate_sigma_hat_prime_sme(tau, self.CL*poi, contrelep1, contrelep2, flavor, Q2, precompute = True, fs = self.fs_vals, fs_prime= self.f_s_prime_vals, num_steps_pdf = self.num_steps_pdf)
        
        ### I will make a right handed version of this plot 
        integral2 = integrate_sigma_hat_prime_sme(tau, self.CR*0, contrelep1, contrelep2, flavor, Q2, precompute = True, fs = fs[i, :], fs_prime= self.f_s_prime_vals, num_steps_pdf = self.num_steps_pdf)
        
        d_sigmaL =  self.sum_terms_L * integral1
        d_sigmaR = self.sum_terms_R * integral2


        return factor * 0.389379 * 1e9*(d_sigmaL + d_sigmaR)  # This is d\sigma / dQ^2
        
    
    ### how do i access the mass
    
'''
so when I call this it will be something like --sme dtdt, dtst blah blah blah

do i want to put all fo them in? 
'''
