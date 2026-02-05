

import time
import pickle

from sme_functions import *


times, pm, pn = get_hour_array() ## i only want to compte this once when i do the 
t = time.time()

mybins = [15, 30, 40, 45, 50, 55, 60, 65, 70, 76, 106, 110, 115, 120]
sme_filename = f"SME_{mybins[0]}_to_{mybins[-1]}_GeV_{len(mybins)-1}_bins.pkl"
sm_filename = f"SM_{mybins[0]}_to_{mybins[-1]}_GeV_{len(mybins)-1}_bins.pkl"

add_dir = "precomputed_sigma/"
precomputed = True
all_precomputed_values = []
all_sm_values = []

SME = True
                
if not SME:
            
    for i in range(len(mybins) - 1):
    # for i in range(1):
        sm = sigma_sm(mybins[i], mybins[i+1], quark_couplings)
        print(sm)
        all_sm_values.append(sm)
    with open(add_dir + sm_filename, "wb") as f:
        output = {"bins": mybins, "values": all_sm_values}
        pickle.dump(output, f)



if SME:
    if precomputed:
        if files_exist([sme_filename, sm_filename]) != 0:
            with open(add_dir + sme_filename, "rb") as f:
                precomp_dict_sme = pickle.load(f)
                if precomp_dict_sme["bins"] != mybins:
                    precomputed = False
        else:
            precomputed = False
    for j in range(len(pm)):     
        for i in range(len(mybins) - 1):
            if precomputed:
                integral_liv = sme(mybins[i], mybins[i+1], pm[j], pn[j], wilson_couplings, precomputed = True, precomputed_values = precomp_dict_sme["values"][i])
            else:
                integral_liv, precomputed_values = sme(mybins[i], mybins[i+1], pm[j], pn[j], wilson_couplings, precomputed = False)
                all_precomputed_values.append(precomputed_values)
    if not precomputed:
        with open(add_dir + sme_filename, "wb") as f:
            output = {"bins": mybins, "values": all_precomputed_values}
            pickle.dump(output, f)
    print(time.time() - t)


