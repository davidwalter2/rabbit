

import time
import pickle

from sme_functions import *


times, pm, pn = get_hour_array() ## i only want to compte this once when i do the 
t = time.time()

mass_bins = [15, 30, 40, 45, 50, 55, 60, 65, 70, 76, 106, 110, 115, 120]
sme_filename = f"SME_{mass_bins[0]}_to_{mass_bins[-1]}_GeV_{len(mass_bins)-1}_bins.pkl"
sm_filename = f"SM_{mass_bins[0]}_to_{mass_bins[-1]}_GeV_{len(mass_bins)-1}_bins.pkl"

add_dir = "precomputed_sigma/"

all_precomputed_values = []
all_sm_values = []

precomputed = False
SME = True
summation = True       
if not SME and not summation:
            
    for i in range(len(mass_bins) - 1):
    # for i in range(1):
        sm = sigma_sm(mass_bins[i], mass_bins[i+1], quark_couplings)
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
    for j in range(len(pm)): #time
        # for i in range(len(mass_bins) - 1):
        for i in range(1):
            if precomputed:
                integral_liv = sme(mass_bins[i], mass_bins[i+1], pm[j], pn[j], wilson_couplings, precomputed = True, precomputed_values = precomp_dict_sme["values"][i])
            else:
                precomputed_values = sme(mass_bins[i], mass_bins[i+1], pm[j], pn[j], wilson_couplings, precomputed = False)
                all_precomputed_values.append(precomputed_values)
    if not precomputed:
        with open(add_dir + sme_filename, "wb") as f:
            output = {"bins": mass_bins, "values": all_precomputed_values}
            pickle.dump(output, f)
    print(time.time() - t)



if summation:
    GeV_to_pb = factor*0.389379*1e9
    n_time_bins = 24
    n_mass_bins = len(mass_bins)
     # sme_filename = f"SME_{self.Q_min}_to_{self.Q_max}_GeV_{len(self.Q_vals)}_bins.pkl" 
    
    sm_filename = f"SM_{min(mass_bins)}_to_{max(mass_bins)}_GeV_{n_mass_bins-1}_bins.pkl" 
    sme_filename = f"SME_{min(mass_bins)}_to_{max(mass_bins)}_GeV_{n_mass_bins-1}_bins.pkl" 
    # sm_filename = "SM_15_to_120_GeV_13_bins.pkl"

    add_dir = "/home/submit/jbenke/WRemnants/rabbit/rabbit/poi_models/precomputed_sigma/"
        
    with open(add_dir + sme_filename, "rb") as f:
        precomp_dict = pickle.load(f)
        # all_precomputed_sme.append(precomp_dict["values"][i]) ## this should only look at the up quark

    precomputed_values = np.array(precomp_dict["values"])

    all_q = []

    for i in range(n_mass_bins-1):
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
    for l in range(len(tensors)):
    # for l in range(1):
        print(f"tensor: {coeff_names[l]}")
        this_CL = tensors[l]
        this_CR = tensors[l]
        for j in range(len(quark_couplings)):
            print(f"quark: {quark[j]}")
        # for j in range(1):
            precomputed_Right = np.zeros([n_time_bins, n_mass_bins])
            precomputed_Left = np.zeros([n_time_bins, n_mass_bins])
            for k in range(n_time_bins-1): #range(24)
                for i in range(n_mass_bins-1):
                # for i in range(1):
                    this_bin = (k*(n_mass_bins-1)) + i
                    ## time (mass)

                    pipi_L_int = GeV_to_pb * precomputed_values[this_bin][:, j, 0] * precomputed_values[this_bin][:, j, 2]
                    pipj_L_int = GeV_to_pb * precomputed_values[this_bin][:,j, 1] * precomputed_values[this_bin][:,j, 2]
                    
                    pipi_R_int = GeV_to_pb *precomputed_values[this_bin][:,j, 3] * precomputed_values[this_bin][:,j, 5]
                    pipj_R_int = GeV_to_pb * precomputed_values[this_bin][:,j, 4] * precomputed_values[this_bin][:,j, 5]
                    
                    integral_pipi_L = simpson(pipi_L_int, Q2_vals[0][i])
                    integral_pipj_L = simpson(pipj_L_int, Q2_vals[0][i])
                    integral_pipi_R = simpson(pipi_R_int, Q2_vals[0][i])
                    integral_pipj_R = simpson(pipj_R_int, Q2_vals[0][i])
                
                    
                    contraction_p1p1_L = tf.einsum('mn,m,n->', this_CL, pm[k], pm[k])
                    contraction_p1p2_L = tf.einsum('mn,m,n->', this_CL, pm[k], pn[k])
                    contraction_p2p1_L = tf.einsum('mn,m,n->', this_CL, pn[k], pm[k])
                    contraction_p2p2_L = tf.einsum('mn,m,n->', this_CL, pn[k], pn[k])
                    
                    contraction_pipi_L = (contraction_p1p1_L + contraction_p2p2_L)
                    contraction_pipj_L = (contraction_p1p2_L + contraction_p2p1_L)
                            
                    contraction_p1p1_R = tf.einsum('mn,m,n->', this_CR, pm[k], pm[k])
                    contraction_p1p2_R = tf.einsum('mn,m,n->', this_CR, pm[k], pn[k])
                    contraction_p2p1_R = tf.einsum('mn,m,n->', this_CR, pn[k], pm[k])
                    contraction_p2p2_R = tf.einsum('mn,m,n->', this_CR, pn[k], pn[k])

                    contraction_pipi_R = (contraction_p1p1_R + contraction_p2p2_R)
                    contraction_pipj_R = (contraction_p1p2_R + contraction_p2p1_R)

                    precomputed_Left[k][i] = integral_pipi_L * contraction_pipi_L + contraction_pipj_L * integral_pipj_L
                    precomputed_Right[k][i] = integral_pipi_R * contraction_pipi_R + integral_pipj_R * contraction_pipj_R

                                        


            filename_L = f"summation_{min(mass_bins)}_to_{max(mass_bins)}_GeV_{n_mass_bins-1}_bins_{coeff_names[l]}_{quark[j]}_L.pkl"
            filename_R = f"summation_{min(mass_bins)}_to_{max(mass_bins)}_GeV_{n_mass_bins-1}_bins_{coeff_names[l]}_{quark[j]}_R.pkl"
            
            
            with open(add_dir + filename_L, "wb") as f:
                output = {"bins": mass_bins, "values": precomputed_Left}
                pickle.dump(output, f)
                
            with open(add_dir + filename_R, "wb") as f:
                output = {"bins": mass_bins, "values": precomputed_Right}
                pickle.dump(output, f)
            
            
            

