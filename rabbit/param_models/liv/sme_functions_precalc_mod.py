### PRECOMPUTE ###
import numpy as np
import torch as tn
from scipy.integrate import quad
import lhapdf
from sme_constants import *
from scipy.integrate import simpson
pdf = lhapdf.mkPDF("NNPDF31_nnlo_as_0118", 0)
from math import pi

from datetime import datetime, timedelta
import pdb
import time
import pickle
import os

    
def get_hour_array(n_time_bins=24):
    """time bin centers in CMS local sidereal hours (the histmaker time axis, 24 bins over
    0-24 h) and the beam momenta pm, pn in the SCF at those times, see sme_constants.scf_beams.
    The beam is horizontal at the CMS azimuth (it was vertical in earlier versions)."""
    hour_array = list((np.arange(n_time_bins) + 0.5) * 24.0 / n_time_bins)
    contrelep1, contrelep2 = scf_beams(hour_array)
    return hour_array, contrelep1, contrelep2
        
##################################################################################
      
def num_derivative(func, x, h=1e-8, *args):
    return (func(x + h, *args) - func(x - h, *args)) / (2 * h)

## inside the integrand

ONE_MINUS = np.nextafter(1.0, 0.0)

def pdf_safe(flavor, x, Q2):
    x = np.minimum(x, ONE_MINUS)
    return pdf.xfxQ2(flavor, x, Q2)


def f_s(x, tau, flavor, Q2):
    tau_x = tau / x

    pdf_flavor_x = pdf_safe(flavor, x, Q2)
    pdf_anti_flavor_x = pdf_safe(-flavor, x, Q2)
    pdf_flavor_tau_x = pdf_safe(flavor, tau_x, Q2)
    pdf_anti_flavor_tau_x = pdf_safe(-flavor, tau_x, Q2)
    term1 = (1 / x) * pdf_flavor_x * (1/tau_x) * pdf_anti_flavor_tau_x
    term2 = (1/tau_x) * pdf_flavor_tau_x * (1 / x) * pdf_anti_flavor_x
    
    return term1 + term2

def f_prime_s(x, tau, flavor, Q2):
    tau_x = tau / x

    # wrap the lambda to clip inside num_derivative
    f_f_tau_x_prime = num_derivative(lambda t: 1/t * pdf_safe(flavor, t, Q2), tau_x)
    f_fbar_tau_x_prime = num_derivative(lambda t: 1/t * pdf_safe(-flavor, t, Q2), tau_x)

    pdf_flavor_x = pdf_safe(flavor, x, Q2)
    pdf_anti_flavor_x = pdf_safe(-flavor, x, Q2)
    
    return (1/x * pdf_flavor_x * f_fbar_tau_x_prime +
            1/x * f_f_tau_x_prime * pdf_anti_flavor_x)
    
    
### THE THREE TERMS IN THE INTEGRAL
def term_1(Q2, e_f):
    return e_f**2 / (2*Q2**2)
    
def term_2(Q2, e_f, g):
    return abs((((1 - (m_Z**2 / Q2)) / ((Q2 - m_Z**2)**2 + m_Z**2 * Gamma_Z**2)) *
            (1 - 4 * sin2th_w) / (4 * sin2th_w * (1- sin2th_w ))* e_f * g))
            
def term_3(Q2, g):
    return (1 / ((Q2 - m_Z**2)**2 + m_Z**2 * Gamma_Z**2) * 
            (1 + (1 - 4 * sin2th_w)**2) / (32 * sin2th_w**2 * (1-sin2th_w)**2)) * g**2

def summation_terms(Q2, e_f, g):
    return  (term_1(Q2, e_f) + term_2(Q2, e_f, g) + term_3(Q2, g))



def integrate_sigma_hat_prime_sme(tau, flavor, Q2):

    def integrand2(x):
        tau_x = tau / x
        term2 = 2*f_s(x, tau, flavor, Q2) + 2* (x**2/tau) * f_s(x, tau, flavor, Q2)
        return term2 * tau_x
    def integrand3(x):
        tau_x = tau / x
        term3 = 2* x * f_prime_s(x, tau, flavor, Q2)
        return term3 * tau_x
    def integrand4(x):
        tau_x = tau / x
        term4 = 2* tau_x * f_prime_s(x, tau, flavor, Q2)
        return term4 * tau_x
    
        
    result2, _ = quad(integrand2, tau, 1)
    result3, _ = quad(integrand3, tau, 1)
    result4, _ = quad(integrand4, tau, 1)

    pipi = (result2 + result3)
    pipj = (result4 + result2) 
    
    return pipi, pipj
    
    


def d_sigma_calc(Q2, p1, p2, quark_couplings):
    tau = Q2 / s
    d_sigmaL = 0
    d_sigmaR = 0
    i = 0
    
    computed_values = []
    for flavor, e_f, g_fR, g_fL, CL, CR in quark_couplings:

        pipi_L, pipj_L = integrate_sigma_hat_prime_sme(tau, flavor, Q2)
        pipi_R, pipj_R = integrate_sigma_hat_prime_sme(tau, flavor, Q2)
        
        sum_terms_L = summation_terms(Q2, e_f, g_fL)
        sum_terms_R = summation_terms(Q2, e_f, g_fR)
        ### currently this only saves the last one
        computed_values.append([pipi_L, pipj_L, sum_terms_L, pipi_R, pipj_R, sum_terms_R])
         
        ### this can't be pulled out further because this iterates by flavor
        contraction_p1p1_L = tn.einsum('mn,m,n->', CL, p1, p1)
        contraction_p1p2_L = tn.einsum('mn,m,n->', CL, p1, p2)
        contraction_p2p1_L = tn.einsum('mn,m,n->', CL, p2, p1)
        contraction_p2p2_L = tn.einsum('mn,m,n->', CL, p2, p2)
        
        contraction_pipi_L = (contraction_p1p1_L + contraction_p2p2_L)
        contraction_pipj_L = (contraction_p1p2_L + contraction_p2p1_L)
        
        
        contraction_p1p1_R = tn.einsum('mn,m,n->', CR, p1, p1)
        contraction_p1p2_R = tn.einsum('mn,m,n->', CR, p1, p2)
        contraction_p2p1_R = tn.einsum('mn,m,n->', CR, p2, p1)
        contraction_p2p2_R = tn.einsum('mn,m,n->', CR, p2, p2)

        contraction_pipi_R = (contraction_p1p1_R + contraction_p2p2_R)
        contraction_pipj_R = (contraction_p1p2_R + contraction_p2p1_R)

        integral1 = pipi_L * contraction_pipi_L + pipj_L* contraction_pipj_L
        integral2 = pipi_R * contraction_pipi_R + pipj_R* contraction_pipj_R
        
        d_sigmaL +=  sum_terms_L * integral1
        d_sigmaR += sum_terms_R * integral2
    
        i += 1 
    return factor * 0.389379 * 1e9*(d_sigmaL + d_sigmaR), computed_values  # This is d\sigma / dQ^2

def d_sigma_precomp(p1, p2, CL, CR, precomputed = False, precomputed_values = []):
    d_sigmaL = 0
    d_sigmaR = 0
    i = 0 #<-- this actu
    
    computed_values = []
    for i in range(len(precomputed_values/6)):
        pipi_L, pipj_L, sum_terms_L, pipi_R, pipj_R, sum_terms_R = precomputed_values[i]
    
         
        ### this can't be pulled out further because this iterates by flavor
        contraction_p1p1_L = tn.einsum('mn,m,n->', CL, p1, p1)
        contraction_p1p2_L = tn.einsum('mn,m,n->', CL, p1, p2)
        contraction_p2p1_L = tn.einsum('mn,m,n->', CL, p2, p1)
        contraction_p2p2_L = tn.einsum('mn,m,n->', CL, p2, p2)
        
        contraction_pipi_L = (contraction_p1p1_L + contraction_p2p2_L)
        contraction_pipj_L = (contraction_p1p2_L + contraction_p2p1_L)
        
        
        contraction_p1p1_R = tn.einsum('mn,m,n->', CR, p1, p1)
        contraction_p1p2_R = tn.einsum('mn,m,n->', CR, p1, p2)
        contraction_p2p1_R = tn.einsum('mn,m,n->', CR, p2, p1)
        contraction_p2p2_R = tn.einsum('mn,m,n->', CR, p2, p2)

        contraction_pipi_R = (contraction_p1p1_R + contraction_p2p2_R)
        contraction_pipj_R = (contraction_p1p2_R + contraction_p2p1_R)

        integral1 = pipi_L * contraction_pipi_L + pipj_L* contraction_pipj_L
        integral2 = pipi_R * contraction_pipi_R + pipj_R* contraction_pipj_R
        
        d_sigmaL +=  sum_terms_L * integral1
        d_sigmaR += sum_terms_R * integral2
    
        i += 1
    if precomputed:
        return factor * 0.389379 * 1e9*(d_sigmaL + d_sigmaR)  # This is d\sigma / dQ^2
    else: 
        return factor * 0.389379 * 1e9*(d_sigmaL + d_sigmaR), computed_values  # This is d\sigma / dQ^2






def sme(Q_min, Q_max, p1, p2, quark_couplings, num_steps_Q2 = 100, precomputed = False, precomputed_values = []):

    if num_steps_Q2 % 2 == 0:
        num_steps_Q2 += 1
    Q2_values = np.linspace(Q_min**2, Q_max**2, num_steps_Q2)
    
    if precomputed:
        integrand_values = np.array([d_sigma_precomp(p1, p2, CL, CR, precomputed = True, precomputed_values = precomputed_values[i]) for i in range(len(Q2_values))])
        
        
    if not precomputed:
        integrand_values, combined_array = zip(*[d_sigma_calc(Q2, p1, p2, quark_couplings) for Q2 in Q2_values])
        ## store left and right in the same file

    # Use Simpson's rule to integrate over the Q2_values array
    integral_liv = simpson(integrand_values, Q2_values)
    print(integral_liv)
    if precomputed:
        return integral_liv
    else:
        return integral_liv, combined_array



times, pm, pn = get_hour_array() ## i only want to compte this once when i do the 
t = time.time()

mybins = [15, 25, 28, 30, 32, 34, 36, 38, 40, 42, 47, 55, 60, 65, 80]
filename = f"{mybins[0]}_to_{mybins[-1]}_GeV_{len(mybins)}_bins.pkl"
add_dir = "precomputed_sme/"
precomputed = True
all_precomputed_values = []

def files_exist(files):
    directory = '.'  # Current directory
    dir_files = os.listdir(directory +f"/{add_dir}")
    return_files = 0

    for file in files:
        if file in dir_files:
        # if os.path.isfile(os.path.join(directory, file)):
            return_files += 1
    if return_files != len(files):
        return 0
    else:
        return files
    
    
    
if precomputed:
    if files_exist([filename]) != 0:
        with open(add_dir + filename, "rb") as f:
            precomp_dict = pickle.load(f)
            if precomp_dict["bins"] != mybins:
                precomputed = False
    else:
        precomputed = False
            
print(precomputed)
# pdb.set_trace()                    
for i in range(len(mybins) - 1):
    if precomputed:
        integral_liv = sme(mybins[i], mybins[i+1], pm[0], pn[0], wilson_couplings, precomputed = True, precomputed_values = precomp_dict["values"][i])
    else:
        integral_liv, precomputed_values = sme(mybins[i], mybins[i+1], pm[0], pn[0], wilson_couplings, precomputed = False)
        all_precomputed_values.append(precomputed_values)
if not precomputed:
    with open(add_dir + filename, "wb") as f:
        output = {"bins": mybins, "values": all_precomputed_values}
        pickle.dump(output, f)
print(time.time() - t)
