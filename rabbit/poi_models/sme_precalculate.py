

import time
import pickle

from sme_functions import *


times, pm, pn = get_hour_array() ## i only want to compte this once when i do the 
t = time.time()

# mass_bins = [15, 30, 40, 45, 50, 55, 60, 65, 70, 76, 106, 110, 115, 120]
mass_bins = [50, 91, 120]
sme_filename = f"SME_{mass_bins[0]}_to_{mass_bins[-1]}_GeV_{len(mass_bins)-1}_bins.pkl"
sm_filename = f"SM_{mass_bins[0]}_to_{mass_bins[-1]}_GeV_{len(mass_bins)-1}_bins.pkl"


n_time_bins = len(pm)
n_mass_bins = len(mass_bins)-1 
    
    
    
add_dir = "precomputed_sigma/"

all_precomputed_values = []
all_sm_values = []

precomputed = False
SME = True
summation = True  
if not SME and not summation:
    for m in range(n_mass_bins):
    # for i in range(9,10):
        print(mass_bins[m])
        print(mass_bins[m+1])
        sm = sigma_sm(mass_bins[m], mass_bins[m+1], quark_couplings)
        print(sm)
        all_sm_values.append(sm)
    with open(add_dir + sm_filename, "wb") as f:
        output = {"bins": mass_bins, "values": all_sm_values}
        pickle.dump(output, f)

if SME and not summation:
    if precomputed:
        if files_exist([sme_filename, sm_filename]) != 0:
            with open(add_dir + sme_filename, "rb") as f:
                precomp_dict_sme = pickle.load(f)
                if precomp_dict_sme["bins"] != mass_bins:
                    precomputed = False
        else:
            precomputed = False
    for t in range(n_time_bins): #time
        for m in range(n_mass_bins):
            if precomputed:
                integral_liv = sme(mass_bins[m], mass_bins[m+1], pm[t], pn[t], wilson_couplings, precomputed = True, precomputed_values = precomp_dict_sme["values"][m])
            else:
                precomputed_values = sme(mass_bins[m], mass_bins[m+1], pm[t], pn[t], wilson_couplings, precomputed = False)
                all_precomputed_values.append(precomputed_values)
    if not precomputed:
        with open(add_dir + sme_filename, "wb") as f:
            output = {"bins": mass_bins, "values": all_precomputed_values}
            pickle.dump(output, f)
    print(time.time() - t)



if summation:
    GeV_to_pb = factor*0.389379*1e9

    sm_filename = f"SM_{min(mass_bins)}_to_{max(mass_bins)}_GeV_{n_mass_bins}_bins.pkl" 
    sme_filename = f"SME_{min(mass_bins)}_to_{max(mass_bins)}_GeV_{n_mass_bins}_bins.pkl" 
    # sm_filename = "SM_15_to_120_GeV_13_bins.pkl"

    add_dir = "/home/submit/jbenke/WRemnants/rabbit/rabbit/poi_models/precomputed_sigma/"
        
    with open(add_dir + sme_filename, "rb") as f:
        precomp_dict = pickle.load(f)


    precomputed_values = np.array(precomp_dict["values"])

    all_q = []

    for i in range(n_mass_bins):
        all_q.append(list(np.linspace(mass_bins[i]**2, mass_bins[i+1]**2, 101)))
        
        
    Q2_vals = np.array([all_q])
    quark = ["u", "d", "s"]
    coeff_names = ["cxx", "cxy", "cxz", "cyz"]
    #precomputed_values[mll][quark]
    #pm[time]
    #pn[time]
    #Q2_vals[time][mll][integration step]
    #precomputed_values[mll][nbins][quark]

    tensors = [CL1, CL2, CL3, CL4]
    ## what changes from quark to quark? 
    for c in range(len(tensors)):
    # for l in range(1):
        print(f"tensor: {coeff_names[c]}")
        this_CL = tensors[c]
        this_CR = tensors[c]
        for q in range(len(quark_couplings)):
            print(f"quark: {quark[q]}")
        # for j in range(1):
            precomputed_Right = np.zeros([n_time_bins, n_mass_bins])
            precomputed_Left = np.zeros([n_time_bins, n_mass_bins])
            for t in range(n_time_bins): 
                for m in range(n_mass_bins):
                    this_bin = (t*(n_mass_bins)) + m
                    pipi_L_int = GeV_to_pb * precomputed_values[this_bin][q, :, 0] * precomputed_values[this_bin][q, :, 2]
                    pipj_L_int = GeV_to_pb * precomputed_values[this_bin][q, :, 1] * precomputed_values[this_bin][q, :, 2]
                    
                    pipi_R_int = GeV_to_pb *precomputed_values[this_bin][q, :, 3] * precomputed_values[this_bin][q, :, 5]
                    pipj_R_int = GeV_to_pb * precomputed_values[this_bin][q, :, 4] * precomputed_values[this_bin][q, :, 5]

                    integral_pipi_L = simpson(pipi_L_int, Q2_vals[0][m])
                    integral_pipj_L = simpson(pipj_L_int, Q2_vals[0][m])
                    integral_pipi_R = simpson(pipi_R_int, Q2_vals[0][m])
                    integral_pipj_R = simpson(pipj_R_int, Q2_vals[0][m])
                
                    
                    contraction_p1p1_L = tf.einsum('mn,m,n->', this_CL, pm[t], pm[t])
                    contraction_p1p2_L = tf.einsum('mn,m,n->', this_CL, pm[t], pn[t])
                    contraction_p2p1_L = tf.einsum('mn,m,n->', this_CL, pn[t], pm[t])
                    contraction_p2p2_L = tf.einsum('mn,m,n->', this_CL, pn[t], pn[t])
                    
                    contraction_pipi_L = (contraction_p1p1_L + contraction_p2p2_L)
                    contraction_pipj_L = (contraction_p1p2_L + contraction_p2p1_L)
                            
                    contraction_p1p1_R = tf.einsum('mn,m,n->', this_CR, pm[t], pm[t])
                    contraction_p1p2_R = tf.einsum('mn,m,n->', this_CR, pm[t], pn[t])
                    contraction_p2p1_R = tf.einsum('mn,m,n->', this_CR, pn[t], pm[t])
                    contraction_p2p2_R = tf.einsum('mn,m,n->', this_CR, pn[t], pn[t])

                    contraction_pipi_R = (contraction_p1p1_R + contraction_p2p2_R)
                    contraction_pipj_R = (contraction_p1p2_R + contraction_p2p1_R)

                    precomputed_Left[t][m] = integral_pipi_L * contraction_pipi_L + contraction_pipj_L * integral_pipj_L
                    precomputed_Right[t][m] = integral_pipi_R * contraction_pipi_R + integral_pipj_R * contraction_pipj_R

                                        


            filename_L = f"summation_{min(mass_bins)}_to_{max(mass_bins)}_GeV_{n_mass_bins}_bins_{coeff_names[c]}_{quark[q]}_L.pkl"
            filename_R = f"summation_{min(mass_bins)}_to_{max(mass_bins)}_GeV_{n_mass_bins}_bins_{coeff_names[c]}_{quark[q]}_R.pkl"
            
            
            with open(add_dir + filename_L, "wb") as f:
                output = {"bins": mass_bins, "values": precomputed_Left}
                pickle.dump(output, f)
                
            with open(add_dir + filename_R, "wb") as f:
                output = {"bins": mass_bins, "values": precomputed_Right}
                pickle.dump(output, f)
            
            
            

