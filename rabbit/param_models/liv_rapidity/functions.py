import numpy as np
import lhapdf
import torch as tn
from datetime import datetime, timedelta, timezone
from scipy.integrate import quad

pdf = lhapdf.mkPDF("NNPDF31_nnlo_as_0118", 0)
S = 13000.0 ** 2
m_Z = 91.1879  # Mass of the Z boson in GeV/c^2
Gamma_Z = 2.4955  # Decay width of the Z boson in GeV
alpha = 0.0072973525643 ### 
sin2th_w = 0.23121  # sin^2(theta_W)
e = 0.3028 # Elementary charge in natural units
SIDEREAL_DAY = 86164 # seconds in one sidereal day

cos2th_w = 1-sin2th_w

def prefactor(Q2):
    return (4 * np.pi * alpha ** 2) / (9 * S * Q2)

def e_f(qid):
    if abs(qid) in (2, 4, 6):
        return 2.0 / 3.0
    else: 
        return -1.0 / 3.0
    
def fx_star(qid, Q2, x1, x2, y, c00, c33, c03):

    fstarx1 = f(qid, x1, Q2) + x1 * delta(1, y, c00, c33, c03) * fprime(qid, x1, Q2)
    antifstarx2 = f(-qid, x2, Q2) + x2 * delta(2, y, c00, c33, c03) * fprime(-qid, x2, Q2)

    fstarx2 = f(qid, x2, Q2) + x2 * delta(2, y, c00, c33, c03) * fprime(qid, x2, Q2)
    antifstarx1 = f(-qid, x1, Q2) + x1 * delta(1, y, c00, c33, c03) * fprime(-qid, x1, Q2)
    return fstarx1 * antifstarx2 + fstarx2 * antifstarx1

def f(qid, x , Q2):
    return pdf.xfxQ2(int(qid), float(x), (Q2)) / x

def fprime(qid, x, Q2):
    h = max(1e-6, x * 1e-4)
    xlo, xhi = x - h, x + h
    if xhi >= pdf.xMax:
        return (f(qid, x, Q2) - f(qid, x - h, Q2)) / h
    if xlo <= pdf.xMin:
        return (f(qid, x + h, Q2) - f(qid, x, Q2)) / h
    return (f(qid, xhi, Q2) - f(qid, xlo, Q2)) / (2.0 * h)

def delta(i, y, c00, c33, c03):
    if i == 1: 
        ret = 0.5 * ((np.exp(-2*y) +  1) * c00 + (np.exp(-2 * y) - 1) * c33 + 2 * np.exp(-2 * y) * c03)
    if i == 2:
        ret = 0.5 * ((np.exp(2*y) +  1) * c00 + (np.exp(2*y) - 1) * c33 - 2 * np.exp(2 * y) * c03)
    return ret

def fwd_bwd_asym(func, y_range):
    mdpt = int(len(func) / 2)
    fwd = np.trapz(func[mdpt:], y_range[mdpt:])
    bwd = np.trapz(func[:mdpt], y_range[:mdpt])
    return fwd - bwd

def total_asym(func, y_range):
    return np.trapz(func, y_range)

def deviation(eq, sm):
    return np.where(sm > 0.0, eq / np.where(sm > 0.0, sm, 1.0) - 1, 0)

def term2(Q2, ef, gf, qid):
    ef = e_f(qid)
    return ((Q2 ** 2)*(1 - (m_Z ** 2 / Q2)) * (1 - 4 * sin2th_w) / 
            (((Q2 - m_Z ** 2) ** 2 + m_Z ** 2 * Gamma_Z ** 2) * 4 * sin2th_w * cos2th_w) * ef * gf)
            
def term3(Q2, gf):
    return ((((Q2 ** 2) * (1 + (1 - 4 * sin2th_w) ** 2) * gf ** 2) / 
             ((((Q2 - m_Z ** 2) ** 2) + m_Z ** 2 * Gamma_Z ** 2) * 32 * sin2th_w ** 2 * cos2th_w ** 2)))

def g_fL(qid):
    ef = e_f(qid)
    if abs(qid) in (2, 4, 6):
        return 0.5 - ef * sin2th_w
    else:
        return -0.5 - ef * sin2th_w
    
def g_fR(qid):
    ef = e_f(qid)
    return -ef * sin2th_w

def _coeff(x):
    return x if hasattr(x, "shape") else np.asarray(x, dtype=float)

def d2sigma_dQ2_dy(qid, Q2, y, cL, cR): #Z and photon
    Q = np.sqrt(Q2)
    if abs(y) > np.log(np.sqrt(S) / Q):
        return 0.0
    tau = float(Q2) / S
    x1 = np.exp(y) * np.sqrt(tau)
    x2 = np.exp(-y) * np.sqrt(tau)
    pref = prefactor(Q2)
    total = 0
    c00L = cL[0] 
    c33L = cL[1]
    c03L = cL[2]

    c00R = cR[0] 
    c33R = cR[1]
    c03R = cR[2]
    for q in qid:
        #Left
        ef = e_f(q)
        term1L = (ef ** 2) / 2
        gfL = g_fL(q)
        term2L = term2(Q2, ef, gfL, q)
        term3L = term3(Q2, gfL)
        FL = fx_star(q, Q2, x1, x2, y, c00L, c33L, c03L)
        total += (term1L + term2L + term3L) * FL
        #Right
        term1R = (ef ** 2) / 2
        gfR = g_fR(q)
        term2R = term2(Q2, ef, gfR, q)
        term3R = term3(Q2, gfR)
        FR = fx_star(q, Q2, x1, x2, y, c00R, c33R, c03R)
        total += (term1R + term2R + term3R) * FR

    return total * pref

#rotations with sidereal time
azimuth = 1.7677    
latitude = 0.8082  
longitude = 0.1061
colatitude = np.pi / 2 - latitude   

# Define the Earth's angular velocity (rad/s)
omega_sidereal = 2*np.pi/(86164)     # Earth's angular velocity in rad/s at UTC.
#omega_siderial = 2*np.pi/(86400)

# time convention of sme_model (liv/sme_functions.py get_hour_array): hour h of the
# time axis is t = SME_T0 + 3600 * h, rotated by omega_sidereal * t + SME_PHASE.
# SME_T0 is datetime(2016, 1, 1, 0, 0).timestamp() as evaluated on the (US/Eastern)
# machine that made the sme_model templates, hardcoded to be timezone independent.
SME_T0 = 1451624400
SME_PHASE = 3.2830

def n_scf(t):
    # beam direction in the SCF, i.e. the spatial part of g R_Z(omega t) R_y(lat) R_z(azi) (0, 0, 0, 1)
    # of sme_model. R_z(azi) leaves the z axis unchanged, so the azimuth drops out.
    phi = omega_sidereal * t + SME_PHASE
    X = -np.cos(latitude) * np.cos(phi)
    Y = -np.cos(latitude) * np.sin(phi)
    Z = -np.sin(latitude)
    return X, Y, Z

def lab_coeffs(t, c):
    nx, ny, nz = n_scf(t)
    c00 = c[0][0]
    c03 = nx * c[0][1] + ny * c[0][2] + nz * c[0][3]
    c33 = (nx ** 2 * c[1][1] + ny ** 2 * c[2][2] + nz ** 2 * c[3][3] + 
           2 * nx * ny * c[1][2] + 2 * nx * nz * c[1][3] + 2 * ny * nz * c[2][3])
    return [c00, c33, c03]

def make_matrix(i, j, val):
    c = tn.tensor([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=tn.float64)

    c[i][j] = val
    c[j][i] = val
    return c

def sidereal_rotation(qid, Q2, cL, cR, y_range, odd):
    hours = np.linspace(0, 24, 25)                   # sidereal hours (closes at 24)
    ts = sidereal_seconds(hours)

    sigsm = np.array([d2sigma_dQ2_dy(qid, Q2, y, [0, 0, 0], [0, 0, 0]) for y in y_range])

    times = hours
    asym = []
    for t in ts:
        clabLV_L = lab_coeffs(t, cL)
        clabLV_R = lab_coeffs(t, cR)
        sigLV = np.array([d2sigma_dQ2_dy(qid, Q2, y, clabLV_L, clabLV_R) for y in y_range])
        sig_deviation = deviation(sigLV, sigsm)

        if odd: 
            asym.append(fwd_bwd_asym(sig_deviation, y_range))
        else:
            asym.append(total_asym(sig_deviation, y_range))
    return asym, times

def integrate_doublexsec_mll(qid, Qmin, Qmax, y, cL, cR):
    def int(Q2):
        return d2sigma_dQ2_dy(qid, Q2, y, cL, cR)
    result, _ = quad(int, Qmin ** 2, Qmax ** 2)
    return result

def integrate_xsec_yll (qid, Qmin, Qmax, ymin, ymax, cL, cR):
    def int(y):
        return integrate_doublexsec_mll(qid, Qmin, Qmax, y, cL, cR)
    result, _ = quad(int, ymin, ymax)
    return result

def dsig_dyll(qid, Q, ymin, ymax, cL, cR):
    def int(y):
        return d2sigma_dQ2_dy(qid, Q ** 2, y, cL, cR)
    result, _ = quad(int, ymin, ymax)
    return result

def get_time():
    hours = np.linspace(0, 24, 25)                   # sidereal hours (closes at 24)
    ts = sidereal_seconds(hours)
    times = []
    times.append(ts)
    return times

def sigma_SCF(qid, Qmin, Qmax, ymin, ymax, cL, cR, t):
    clabLV_L = lab_coeffs(t, cL)
    clabLV_R = lab_coeffs(t, cR)
    return integrate_xsec_yll(qid, Qmin, Qmax, ymin, ymax, clabLV_L, clabLV_R)

def sidereal_seconds(hour, n_time_bins = 24):
    # same as sme_model: one bin per 3600 s starting at SME_T0
    return SME_T0 + hour * (24 * 3600 / n_time_bins)


def d2sigma_p(qid, Q2, y, p1, p2, cL, cR):
    Q = np.sqrt(Q2)
    if abs(y) > np.log(np.sqrt(S) / Q):
        return 0.0
    tau = float(Q2) / S
    x1 = np.exp(y) * np.sqrt(tau)
    x2 = np.exp(-y) * np.sqrt(tau)
    pref = prefactor(Q2)
    total = 0
    for q in qid:
        #Left
        ef = e_f(q)
        term1L = (ef ** 2) / 2
        gfL = g_fL(q)
        term2L = term2(Q2, ef, gfL, q)
        term3L = term3(Q2, gfL)
        
        total += (term1L + term2L + term3L) * FL
        #Right
        term1R = (ef ** 2) / 2
        gfR = g_fR(q)
        term2R = term2(Q2, ef, gfR, q)
        term3R = term3(Q2, gfR)

        total += (term1R + term2R + term3R) * FR

    return total * pref

def fx_star_p(qid, Q2, x1, x2, y, c, p1, p2):

    fstarx1 = f(qid, x1, Q2) + x1 * delta(1, y, c00, c33, c03) * fprime(qid, x1, Q2)
    antifstarx2 = f(-qid, x2, Q2) + x2 * delta(2, y, c00, c33, c03) * fprime(-qid, x2, Q2)

    fstarx2 = f(qid, x2, Q2) + x2 * delta(2, y, c00, c33, c03) * fprime(qid, x2, Q2)
    antifstarx1 = f(-qid, x1, Q2) + x1 * delta(1, y, c00, c33, c03) * fprime(-qid, x1, Q2)
    return fstarx1 * antifstarx2 + fstarx2 * antifstarx1

def delta_p(i, p1, p2, c, x1, x2):
    if i == 1: 
        ret = 0.5 * ((np.exp(-2*y) +  1) * c00 + (np.exp(-2 * y) - 1) * c33 + 2 * np.exp(-2 * y) * c03)
    if i == 2:
        ret = 0.5 * ((np.exp(2*y) +  1) * c00 + (np.exp(2*y) - 1) * c33 - 2 * np.exp(2 * y) * c03)
    return ret