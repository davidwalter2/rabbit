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
# Sign convention (shared with delta(), i.e. Eqs. 17-19 of the note):
#   * the lab "3" axis is the experiment's +z axis, i.e. the direction of positive
#     (pseudo)rapidity of the reconstructed leptons (eta_probe / yll);
#   * parton 1 (x1 = sqrt(tau) e^{+y}) moves along +z, parton 2 along -z;
#   * c03 = zhat^i c^{0i}, c33 = zhat^i zhat^j c^{ij}, with zhat the +z unit vector in the
#     SCF. Spatial rotations act identically on upper and lower spatial indices, so the
#     SCF coefficients c[0][i], c[i][j] carry the same index position as the paper's lab
#     c^{03}, c^{33} (no metric factor is applied).
# Location and +z azimuth (measured from north towards east) of the experiments. Both
# define x towards the ring centre, y up and z = x cross y along the anticlockwise beam
# (seen from above). CMS (Point 5) is on the north side of the ring, so its +z points
# roughly west, towards the Jura (281.3 deg); ATLAS (Point 1) is diametrically opposite
# and its +z points roughly east, towards the airport (101.3 deg; cf. Carle, Chanon,
# Perries, arXiv:1909.01990, who quote the 101.28 deg beam line). Only the c03-type
# (ctx, cty) terms, linear in zhat, change sign between the two; the latitude differs by
# 0.07 deg.
EXPERIMENTS = {
    "CMS": dict(latitude=0.8082, azimuth=1.7677 + np.pi, longitude=0.1061),
    "ATLAS": dict(
        latitude=np.deg2rad(46.2358), azimuth=1.7677, longitude=np.deg2rad(6.0554)
    ),
}
azimuth = EXPERIMENTS["CMS"]["azimuth"]
latitude = EXPERIMENTS["CMS"]["latitude"]
longitude = EXPERIMENTS["CMS"]["longitude"]
colatitude = np.pi / 2 - latitude   

def get_experiment(experiment):
    try:
        return EXPERIMENTS[experiment.upper()]
    except KeyError:
        raise ValueError(f"unknown experiment {experiment}, choose from {list(EXPERIMENTS)}")

# time is the local sidereal time at the experiment in sidereal hours, as filled by the
# histmakers (GMST from J2000 plus the experiment longitude). The rotation from the lab
# frame to the SCF is done here: the angle of the local meridian from the vernal equinox
# is 2 pi LST / 24, so the longitude must not be added again.
def sidereal_angle(lst):
    return 2 * np.pi * np.asarray(lst) / 24.0

def n_scf(lst, experiment="CMS"):
    # unit vector zhat of the experiment's +z axis in the SCF (components zhat^i),
    # zhat = R_Z(phi) R_y(colatitude) zhat_lab. The local frame is (south, east, up), and
    # +z is horizontal at an azimuth measured from north towards east:
    # zhat_lab = (-cos(azi), sin(azi), 0).
    loc = get_experiment(experiment)
    lat, azi = loc["latitude"], loc["azimuth"]
    phi = sidereal_angle(lst)
    # R_y(colatitude) zhat_lab, the local meridian at phi = 0
    x = -np.sin(lat) * np.cos(azi)
    y = np.sin(azi)
    z = np.cos(lat) * np.cos(azi)
    # R_Z(phi), the earth rotation
    X = x * np.cos(phi) - y * np.sin(phi)
    Y = x * np.sin(phi) + y * np.cos(phi)
    Z = z + 0 * phi
    return X, Y, Z

def lab_coeffs(lst, c, experiment="CMS"):
    nx, ny, nz = n_scf(lst, experiment)
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

    sigsm = np.array([d2sigma_dQ2_dy(qid, Q2, y, [0, 0, 0], [0, 0, 0]) for y in y_range])

    times = hours
    asym = []
    for t in hours:
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
    times = []
    times.append(hours)
    return times

def sigma_SCF(qid, Qmin, Qmax, ymin, ymax, cL, cR, t):
    clabLV_L = lab_coeffs(t, cL)
    clabLV_R = lab_coeffs(t, cR)
    return integrate_xsec_yll(qid, Qmin, Qmax, ymin, ymax, clabLV_L, clabLV_R)


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