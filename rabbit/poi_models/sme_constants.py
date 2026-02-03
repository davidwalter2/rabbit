import torch as tn
import numpy as np
# Constants
s = (13e3)**2 # Center-of-mass energy squared in GeV^2
Nc = 3  # Number of colors in QCD
m_Z = 91.1876  # Mass of the Z boson in GeV/c^2
Gamma_Z = 2.4952  # Decay width of the Z boson in GeV
alpha = 1/137
sin2th_w = 0.23121  # sin^2(theta_W)
e = 0.3028 # Elementry charge in natural units

quarks = [
    (2, 2/3*e, 'u', 1/2),
    (1, -1/3*e, 'd', -1/2),
    (3, -1/3*e, 's', -1/2),
    #  (4, 2/3*e, 'c', 1/2),
    #   (5, -1/3*e, 'b', -1/2),
    #  (6, 2/3*e, 't', 1/2),
]


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

CL1 = tn.tensor([
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, -1, 0],
    [0,0, 0, 0]
], dtype=tn.float32)

CL4 = tn.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, -1],
    [0,0,-1, 0]
], dtype=tn.float32)


CL3 = tn.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, -1],
    [0, 0, 0, 0],
    [0,-1, 0, 0]
], dtype=tn.float32)
# cl_4_coeff = 1.22031
CL4 = tn.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, -1],
    [0,0,-1, 0]
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



wilson_coeffs_L = 1e-5*np.ones((len(quarks)))
wilson_coeffs_R = 0*np.ones((len(quarks)))

wilson_couplings = []
i = 0
for flavor, e_f, name, I3 in quarks:
    g_fR = -e_f * sin2th_w
    g_fL = I3 - e_f * sin2th_w
    
    # Rounding to 4 decimal places
    e_f = round(e_f, 10)
    g_fR = round(g_fR, 10)
    g_fL = round(g_fL, 10)
    
    c_fL = wilson_coeffs_L[i]
    c_fR = wilson_coeffs_R[i]

    i += 1
    ### i'll need to switch this to also iterate through the different c matricies
    wilson_couplings.append((flavor, e_f, g_fR, g_fL, c_fL*CL4, c_fR*CR))



