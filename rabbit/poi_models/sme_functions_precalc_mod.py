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

    
### i may want to do this as an array of times
def get_hour_array():
    ### expects input of the form: specific_time = datetime(2017, 1, 1, 0, 0)
    # Define the location of CMS in terms of longitude, latitude and azimuth
    azimuth = 1.7677    
    latitude = 0.8082  
    longitude = 0.1061   

    # Define the Earth's angular velocity (rad/s)
    omega_utc = 2*pi/(86164)     # Earth's angular velocity in rad/s at UTC.
    omega_siderial = 2*pi/(86400)
    # Rotation matrices to go from SCF to CMS frame

    # Rotation around the z-axis by phi (due to the azimuthal angle in spherical coordinates):
    def R_z(angle):
        return tn.tensor([
            [1, 0, 0, 0],
            [0,  np.cos(angle), -np.sin(angle),0],
            [0, np.sin(angle),  np.cos(angle),0],
            [0,0,0,1]
        ], dtype=tn.float32)

    # Rotation around the y-axis by θ (aligning the z-axis with the polar axis):
    def R_y(angle):
        return tn.tensor([
            [1, 0, 0, 0],
            [0, np.sin(angle), 0, np.cos(angle)],
            [0, 0, 1, 0],
            [0, -np.cos(angle), 0, np.sin(angle)]
        ], dtype=tn.float32)

    # A final rotation around the Z-axis has two purposes: to follow the rotation of the Earth over time and to synchronize with the SCF:
    def R_Z(angle):
        return tn.tensor([
            [1, 0 , 0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1]
        ], dtype=tn.float32)



    specific_time = datetime(2017, 1, 1, 0, 0)

    start_time = int(specific_time.timestamp())

    # start_time = int(time.time())
    end_time = start_time + 3600 #+ int(timedelta(days=1).total_seconds())
    step_seconds = int(timedelta(hours=1).total_seconds())
    num_steps = (end_time - start_time) // step_seconds

    # Lists to store the times and contr matrix elements
    times = []
    contrelep1 = []
    contrelep2 = []

    R_y_lat = R_y(latitude)
    R_z_azi = R_z(azimuth)
    mat_cons = tn.matmul(R_y_lat,R_z_azi)
    # Main loop
    current_time = start_time
    for _ in range(num_steps):
        # Convert current_time to a timestamp
        current_datetime = datetime.fromtimestamp(current_time)
        time_utc = current_datetime.timestamp()

        # Calculate omega_t
        omega_t_sid = omega_utc * time_utc + 3.2830 
        # Construct the complete rotation matrix from SCF to CMS
        R_Z_omega = R_Z(omega_t_sid)
        R_mat = tn.matmul(R_Z_omega, mat_cons)
        R_matrix1 = tn.einsum('ma,an->mn', g, R_mat)
        R_matrix2 = tn.einsum('am,na->mn', g, R_mat)
        # print(R_matrix1)
        # Compute contrL and contrR using matrix multiplication
        contrp1 = tn.einsum('ij,j->i', R_matrix1, p1)
        contrp2 =  tn.einsum('ij,i->j',R_matrix2, p2)
        # Record the times and contr matrix elements
        times.append(current_time)
        contrelep1.append(contrp1)
        contrelep2.append(contrp2)
        # Move to the next time step
        current_time += step_seconds
    hour_array = [(t - start_time) / 3600 for t in times]
    return hour_array, contrelep1, contrelep2   # Convert seconds to hours 
        
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
    
    


def d_sigma(Q2, p1, p2, quark_couplings, precomputed = False, Q_min = 0, Q_max = 0, num_steps = 100):
    tau = Q2 / s
    d_sigmaL = 0
    d_sigmaR = 0
    i = 0
    
    for flavor, e_f, g_fR, g_fL, CL, CR in quark_couplings:
        if precomputed:
            filename = f"precomputed_sme/{Q_min}_to_{Q_max}_{Q2}GeV_{flavor}_{num_steps}.pkl"
            try:
                with open(filename, "rb") as f:
                    combined_array = pickle.load(f)
                    pipi_L, pipj_L, sum_terms_L, pipi_R, pipj_R, sum_terms_R = combined_array
            except: 
                precomputed = False
        if not precomputed:
            pipi_L, pipj_L = integrate_sigma_hat_prime_sme(tau, flavor, Q2)
            pipi_R, pipj_R = integrate_sigma_hat_prime_sme(tau, flavor, Q2)
            
            sum_terms_L = summation_terms(Q2, e_f, g_fL)
            sum_terms_R = summation_terms(Q2, e_f, g_fR)
        
            
            filename = f"precomputed_sme/{Q_min}_to_{Q_max}_{Q2}GeV^2_{quarks[flavor-1]}_{num_steps}_steps.pkl"
            ## store left and right in the same file
            combined_array = np.array([pipi_L,pipj_L, sum_terms_L, pipi_R, pipj_R,  sum_terms_R])
            with open(filename, "wb") as f:
                pickle.dump(combined_array, f)
                
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

    return factor * 0.389379 * 1e9*(d_sigmaL + d_sigmaR)  # This is d\sigma / dQ^2



def sme(Q_min, Q_max, p1, p2, quark_couplings, num_steps_Q2 = 100, precomputed = False):

    num_steps_Q2 = int(num_steps_Q2)
    if num_steps_Q2 % 2 == 0:
        num_steps_Q2 += 1
    Q2_values = np.linspace(Q_min**2, Q_max**2, num_steps_Q2)

    integrand_values = np.array([d_sigma(Q2, p1, p2, quark_couplings,precomputed, Q_min, Q_max) for Q2 in Q2_values])
    
    # Use Simpson's rule to integrate over the Q2_values array
    integral_liv = simpson(integrand_values, Q2_values)
    print(integral_liv)
    return integral_liv



times, pm, pn = get_hour_array() ## i only want to compte this once when i do the 
t = time.time()
# sme(70, 80, pm[0], pn[0], wilson_couplings)

mybins = [15, 25, 28, 30, 32, 34, 36, 38, 40, 42, 47, 55, 60, 65, 80]
# for i in range(len(mybins) - 1):
for i in range(0, 1):
    sme(mybins[i], mybins[i+1], pm[0], pn[0], wilson_couplings)
print(t - time.time())
