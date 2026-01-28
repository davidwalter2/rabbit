import numpy as np
import tensorflow as tf
from sme_constants import *
from sme_functions import *

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
        num_steps_pdf = 200,
        num_steps_mass_bin = 100
    ):
        self.npoi = npoi
        self.pois = poi_names
        self.xpoidefault = poi_defaults
        self.is_linear = False
        self.allowNegativePOI = False
            
        

        self.CL = left_tensor
        self.CR = right_tensor
        self.num_steps_pdf = num_steps_pdf
        
        self.Q_vals = indata.axes
        
        ### need to have the mass bins for this and the binning
        self.fs_vals, self.fs_prime_vals = precompute_fs(Qmin, Qmax, num_steps_pdf, True)  ### this will return the values 
        
        self.sum_terms_L = summation_terms(Q2, e_f, g_fL)
        self.sum_terms_R = summation_terms(Q2, e_f, g_fR)
        
        
        

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
