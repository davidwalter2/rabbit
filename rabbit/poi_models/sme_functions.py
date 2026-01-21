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



factor = 4 * alpha**2*np.pi / (3 * Nc)

g = tn.tensor([
    [1,0,0,0],
    [0,-1,0,0],
    [0,0,-1,0],
    [0,0,0,-1]
], dtype=tn.float32)


p1 =  0.5*tn.tensor([1, 0, 0, 1], dtype=tn.float32)
p2 =  0.5*tn.tensor([1, 0, 0, -1], dtype=tn.float32)


CR = tn.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
], dtype=tn.float32)



cl_1_coeff = 1e-5
cl_4_coeff = 1e-5

CL1 = tn.tensor([
    [0, 0, 0, 0],
    [0, cl_1_coeff, 0, 0],
    [0, 0, -cl_1_coeff, 0],
    [0,0, 0, 0]
], dtype=tn.float32)

CL4 = tn.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, -cl_4_coeff],
    [0,0,-cl_4_coeff, 0]
], dtype=tn.float32)


quark_couplings = []
for flavor, e_f, name, I3 in quarks:
    g_fR = -e_f * sin2th_w
    g_fL = I3 - e_f * sin2th_w
    
    # Rounding to 4 decimal places
    e_f = round(e_f, 10)
    g_fR = round(g_fR, 10)
    g_fL = round(g_fL, 10)
    
    quark_couplings.append((flavor, e_f, g_fR, g_fL))
    

    
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
            
def term_3(Q2, e_f, g):
    return (1 / ((Q2 - m_Z**2)**2 + m_Z**2 * Gamma_Z**2) * 
            (1 + (1 - 4 * sin2th_w)**2) / (32 * sin2th_w**2 * (1-sin2th_w)**2)) * g**2

def summation_terms(Q2, e_f, g):
    return  (term_1(Q2, e_f) + term_2(Q2, e_f, g) + term_3(Q2, e_f, g))

##########################################################
# STANDARD MODEL CALCULATION #

def integrate_sigma_hat_prime_sm(tau, flavor, Q2):
    def integrand1(x):
        tau_x = tau/x
        return f_s(x, tau, flavor, Q2) * tau_x
    
    result1, _ = quad(integrand1, tau, 1)
    return result1

def d_sigma_sm(Q2, quark_couplings):
    tau = Q2 / s 
    d_sigma = 0
    for flavor, e_f, g_fR, g_fL in quark_couplings:
        integral = integrate_sigma_hat_prime_sm(tau, flavor, Q2)
        termL = summation_terms(Q2, e_f, g_fL)
        termR = summation_terms(Q2, e_f, g_fR)
        d_sigma +=  (termL+ termR )* integral
    
    d_sigmasm =  factor * 0.389379 * 1e9* d_sigma    # Conversion from GeV^-2 to Pb
    return d_sigmasm

def sigma_sm(Qmin, Qmax, quark_couplings):
    def inet(Q2):
        return d_sigma_sm(Q2, quark_couplings)
    int, _ = quad(inet, Qmin**2, Qmax**2)
    return int

def get_sm_sigma(Q_vals): ### pass in a list of maxx bins
    
    Q_min = [Q_vals[i] for i in range(len(Q_vals) -1)]
    Q_max = [Q_vals[i] for i in range(1, len(Q_vals))]
    # create the keys those bins would correspond to 
    sample_keys = [f"{Q_min[i]}_{Q_max[i]}" for i in range(len(Q_min))]
    try: # if there is this file pickled, load it
        with open("precomputed_sm_sigma.pkl", "rb") as f:
            precomputed_sm_sigma = pickle.load(f)
            print("LOADED PRECOMPUTED SIGMA_SM")
    except: # if the file doesn't exist, create a blank dictionary (only needed the very first time)
        precomputed_sm_sigma = {}
    precomputed_sm_sigma_keys = precomputed_sm_sigma.keys()
    # compare to see if all the necessary keys were already computed
    precomputed = [sample_keys[i] in precomputed_sm_sigma_keys for i in range(len(sample_keys))]
    
    if False in precomputed: 
        for i in range(len(precomputed)):
            if i == False:
                sigma_val = sigma_sm(Q_min[i], Q_max[i], quark_couplings)
                for i in range(len(Q_min)):
                    precomputed_sm_sigma[f"{Q_min[i]}_{Q_max[i]}"] = sigma_val
            with open("precomputed_sm_sigma.pkl", "wb") as f:
                pickle.dump(precomputed_sm_sigma, f)
    return [precomputed_sm_sigma[this_key] for this_key in sample_keys]



###############################################################################3
### STANDARD MODEL EXTENSION


def sigma_hat_prime(x, tau, C, p1, p2, flavor, Q2):
    # try:
    tau_x = tau/x
    
    f_s_val = f_s(x, tau, flavor, Q2)
    f_prime_s_val = f_prime_s(x, tau, flavor, Q2)
        
        ### need to save this
    # Efficiently handle the contraction with non-zero elements of C
    contraction_p1p1 = tn.einsum('mn,m,n->', C, p1, p1)
    contraction_p1p2 = tn.einsum('mn,m,n->', C, p1, p2)
    contraction_p2p1 = tn.einsum('mn,m,n->', C, p2, p1)
    contraction_p2p2 = tn.einsum('mn,m,n->', C, p2, p2)
    
    term1 = f_s_val
    term2 = 2* (1 + x / tau_x) * (contraction_p1p1 + contraction_p1p2 +  contraction_p2p1 + contraction_p2p2) * f_s_val
    term3 = 2 * (x * contraction_p1p1 +  tau_x * contraction_p1p2 + tau_x * contraction_p2p1 + x * contraction_p2p2) * f_prime_s_val

    return term1, term2 + term3

def integrate_sigma_hat_prime_sme(tau, C, p1, p2, flavor, Q2, precompute = False, fs = [], fs_prime = [], num_steps_pdf = 0):
    def integrand2(x):
        tau_x = tau / x
        _, term2_plus_term3 = sigma_hat_prime(x, tau, C, p1, p2, flavor, Q2)
        return term2_plus_term3 * tau_x
    
    def integrand2_precomputed(x, fs, fs_prime, contraction_p1p1, contraction_p1p2, contraction_p2p1, contraction_p2p2):
        tau_x = tau / x
        term2 = 2* (1 + x / tau_x) * (contraction_p1p1 + contraction_p1p2 +  contraction_p2p1 + contraction_p2p2) * fs
        term3 = 2 * (x * contraction_p1p1 +  tau_x * contraction_p1p2 + tau_x * contraction_p2p1 + x * contraction_p2p2) * fs_prime
        return (term2 + term3) * tau_x
    # pdb.set_trace()
    
    eps = 1e-12
    integration_bounds = np.geomspace(tau / (1 - eps), 1 - eps, num_steps_pdf)
    
    if precompute:
        contraction_p1p1 = tn.einsum('mn,m,n->', C, p1, p1)
        contraction_p1p2 = tn.einsum('mn,m,n->', C, p1, p2)
        contraction_p2p1 = tn.einsum('mn,m,n->', C, p2, p1)
        contraction_p2p2 = tn.einsum('mn,m,n->', C, p2, p2)
        # pdb.set_trace() ### PROBLEM HERE IS THAT TAU ALREADY HAS A SHAPE
        y = [integrand2_precomputed(integration_bounds[i], fs[i], fs_prime[i], contraction_p1p1, contraction_p1p2, contraction_p2p1, contraction_p2p2) for i in range(integration_bounds.shape[0])]
        result2 = simpson(y, x = integration_bounds)
    
    else: 
        y = [integrand2(x) for x in integration_bounds]
        result2, _ = quad(integrand2, tau, 1)

    return result2

def d_sigma(Q2, CL, CR, p1, p2, quark_couplings, precompute = False, fs = [], fs_prime = [], num_steps_pdf = 0):
    tau = Q2 / s
    d_sigmaL = 0
    d_sigmaR = 0
    i = 0
    for flavor, e_f, g_fR, g_fL in quark_couplings:
        if precompute:
            integral1 = integrate_sigma_hat_prime_sme(tau, CL, p1, p2, flavor, Q2, precompute = precompute, fs = fs[i, :], fs_prime= fs_prime[i, :], num_steps_pdf = num_steps_pdf)
            integral2 = integrate_sigma_hat_prime_sme(tau, CR, p1, p2, flavor, Q2, precompute = precompute, fs = fs[i, :], fs_prime= fs_prime[i, :], num_steps_pdf = num_steps_pdf)
        else:
            integral1 = integrate_sigma_hat_prime_sme(tau, CL, p1, p2, flavor, Q2)
            integral2 = integrate_sigma_hat_prime_sme(tau, CR, p1, p2, flavor, Q2)
        
        sum_terms_L = summation_terms(Q2, e_f, g_fL)
        sum_terms_R = summation_terms(Q2, e_f, g_fR)

        d_sigmaL +=  sum_terms_L * integral1
        d_sigmaR += sum_terms_R * integral2
        
        i += 1

    return factor* 0.389379 * 1e9*(d_sigmaL + d_sigmaR)  # This is d\sigma / dQ^2


def precompute_fs(Q_min, Q_max, num_steps_Q2, num_steps_PDF):

    Q2_values = np.linspace(Q_min**2, Q_max**2, num_steps_Q2)
    file_name = f"{Q_min}_to_{Q_max}_GeV_{num_steps_Q2}_Q2_steps_{num_steps_PDF}_PDF_steps" # not labeling the quark coupling because i think we only intend on doing up, down, strange
    f_s_all = np.zeros((len(quark_couplings), num_steps_Q2, num_steps_PDF))
    f_prime_s_all = np.zeros((len(quark_couplings), num_steps_Q2, num_steps_PDF))   # quark, Q2, momentum fraction
    print(len(quark_couplings))
    k = 0 # to keep track of which quark coupling we are looking at. 
    for flavor, _, _, _ in quark_couplings:
        for i in range(Q2_values.shape[0]):
            Q2 = Q2_values[i]
            tau = Q2 / s
            eps = 1e-12
            x = np.geomspace(tau / (1 - eps), 1 - eps, num_steps_PDF)
            for j in range(x.shape[0]):  
                f_s_all[k, i, j] = f_s(x[j], tau, flavor, Q2)
                f_prime_s_all[k, i, j] = f_prime_s(x[j], tau, flavor, Q2)
        k += 1
    with open(f"fs_{file_name}.pkl", "wb") as f:
        pickle.dump(f_s_all, f)
    with open(f"f_prime_s_{file_name}.pkl", "wb") as f:
        pickle.dump(f_prime_s_all, f)
    
    return

def files_exist(files):
    directory = '.'  # Current directory
    dir_files = os.listdir(directory)
    return_files = 0
    for file in files:
        if file in dir_files:
        # if os.path.isfile(os.path.join(directory, file)):
            return_files += 1
    if return_files != len(files):
        return 0
    else:
        return files

def sme(Q_min, Q_max, CL, CR, p1, p2, quark_couplings, num_steps_Q2 =100, precompute = False, num_steps_PDF = 200):
    num_steps_Q2 = int(num_steps_Q2)
    num_steps_PDF = int(num_steps_PDF)
    # Ensure that num_steps is odd for Simpson's rule to work properly
    if num_steps_Q2 % 2 == 0:
        num_steps_Q2 += 1
    if num_steps_PDF % 2 == 0:
        num_steps_PDF += 1
    Q2_values = np.linspace(Q_min**2, Q_max**2, num_steps_Q2)
    
    # Loop over Q2 values and calculate d_sigma for each
    if precompute:
        file_name = f"{Q_min}_to_{Q_max}_GeV_{num_steps_Q2}_Q2_steps_{num_steps_PDF}_PDF_steps"
        files = files_exist([f"fs_{file_name}.pkl", f"f_prime_s_{file_name}.pkl"])
        if type(files) == int:
            print("FS OR FS' HAS NOT BEEN PREVIOUSLY COMPUTED. COMPUTING NOW")
            precompute_fs(Q_min, Q_max, num_steps_Q2 = num_steps_Q2, num_steps_PDF = num_steps_PDF)
            
        with open(f"fs_{file_name}.pkl", "rb") as f:
            fs = pickle.load(f)
        with open(f"f_prime_s_{file_name}.pkl", "rb") as f:
            fs_prime = pickle.load(f)
        integrand_values = np.array([d_sigma(Q2_values[i], CL, CR, p1, p2, quark_couplings, precompute = precompute, fs = fs[:, i, :], fs_prime = fs_prime[:, i, :], num_steps_pdf=num_steps_PDF) for i in range(Q2_values.shape[0])])
    else:
        integrand_values = np.array([d_sigma(Q2, CL, CR, p1, p2, quark_couplings) for Q2 in Q2_values])
    
    # Use Simpson's rule to integrate over the Q2_values array
    integral_liv = simpson(integrand_values, Q2_values)
    print(integral_liv)
    return integral_liv

times, pm, pn = get_hour_array() ## i only want to compte this once when i do the 
t = time.time()
sme(70, 80, CL4, CR, pm[0], pn[0], quark_couplings, precompute = True)
print(t - time.time())
