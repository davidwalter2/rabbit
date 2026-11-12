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
    (2, 2/3, 'u', 1/2),
    (1, -1/3, 'd', -1/2),
    (3, -1/3, 's', -1/2),
    #  (4, 2/3, 'c', 1/2),
    #   (5, -1/3, 'b', -1/2),
    #  (6, 2/3, 't', 1/2),
]


factor = 4 * alpha**2*np.pi / (3 * Nc)

g = tn.tensor([
    [1,0,0,0],
    [0,-1,0,0],
    [0,0,-1,0],
    [0,0,0,-1]
], dtype=tn.float32)


# CMS location and LHC beam direction
azimuth = 1.7677  # beam azimuth, measured from north towards east [rad]
latitude = 0.8082  # [rad]

# beam momenta in the local frame (south, east, up). The beam is horizontal at the
# azimuth above, n_lab = (-cos(azi), sin(azi), 0). NB 101.28 deg is the clockwise beam,
# i.e. CMS -z (CMS +z points towards the Jura, 281.28 deg); scf_beams lowers the index
# with g, which flips the spatial part, so the covariant pm has spatial part along CMS +z.
# liv_rapidity/functions.py n_scf uses the +z azimuth directly and no g (same numbers).
n_lab = (-np.cos(azimuth), np.sin(azimuth), 0.0)
p1 = 0.5 * tn.tensor([1, n_lab[0], n_lab[1], n_lab[2]], dtype=tn.float32)
p2 = 0.5 * tn.tensor([1, -n_lab[0], -n_lab[1], -n_lab[2]], dtype=tn.float32)


def scf_beams(hours):
    """contravariant beam momenta pm, pn [hour][4] in the SCF at the CMS local sidereal
    times `hours` (sidereal hours, the time axis of the histmakers, which already include
    the CMS longitude). The CMS meridian is at the angle 2 pi LST / 24 from the vernal
    equinox: p_SCF = R_Z(2 pi LST / 24) R_y(latitude) p_lab, index lowered with g."""

    def R_y(angle):
        return tn.tensor([
            [1, 0, 0, 0],
            [0, np.sin(angle), 0, np.cos(angle)],
            [0, 0, 1, 0],
            [0, -np.cos(angle), 0, np.sin(angle)]
        ], dtype=tn.float32)

    def R_Z(angle):
        return tn.tensor([
            [1, 0, 0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1]
        ], dtype=tn.float32)

    R_y_lat = R_y(latitude)
    pm, pn = [], []
    for h in hours:
        R = tn.matmul(R_Z(2 * np.pi * float(h) / 24.0), R_y_lat)
        pm.append(tn.einsum("ma,an,n->m", g, R, p1))
        pn.append(tn.einsum("ma,an,n->m", R, g, p2))
    return pm, pn


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

CL2 = tn.tensor([
    [0, 0, 0, 0],
    [0, 0, -1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 0]
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



# quarks summed in the SM cross section (the denominator of the relative SME shift):
# also c, b, t, which have no SME templates (the 5 flavour PDF set has no top, which adds zero)
sm_quarks = quarks + [
    (4, 2/3, 'c', 1/2),
    (5, -1/3, 'b', -1/2),
    (6, 2/3, 't', 1/2),
]
sm_quark_couplings = []
for flavor, e_f, name, I3 in sm_quarks:
    g_fR = -e_f * sin2th_w
    g_fL = I3 - e_f * sin2th_w
    sm_quark_couplings.append((flavor, round(e_f, 10), round(g_fR, 10), round(g_fL, 10)))


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
    wilson_couplings.append((flavor, e_f, g_fR, g_fL, CL1, CR))



