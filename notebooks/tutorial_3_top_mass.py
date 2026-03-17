import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import hist
import matplotlib.pyplot as plt
import mplhep as hep
from rabbit import tensorwriter

# ==============================================================================
# Tutorial 3: Fitting the Top Quark Mass with in-situ JES calibration
# ==============================================================================
# In this tutorial we extract the top quark mass from the invariant mass
# distribution of the 3-jet combination (reconstructed top) while simultaneously
# fitting the Jet Energy Scale (JES) using the 2-jet combination (reconstructed W).
#
# We use a 2D template approach (m_W vs m_top) to capture the correlation between
# the two observables. The JES acts as a constrained shape systematic that shifts
# both m_W and m_top. The top mass shift acts as an unconstrained parameter
# affecting only m_top.

# ------------------------------------------------------------------------------
# Setting up the environment
# ------------------------------------------------------------------------------
rabbit_base = str(Path(os.getcwd()).parent.absolute())
os.environ['RABBIT_BASE'] = rabbit_base

if rabbit_base not in sys.path:
    sys.path.append(rabbit_base)

rabbit_bin = os.path.join(rabbit_base, 'bin')
current_path = os.environ.get('PATH', '')
if rabbit_bin not in current_path:
    os.environ['PATH'] = f"{current_path}:{rabbit_bin}"

# ------------------------------------------------------------------------------
# Generating the invariant mass distributions (2D: m_W vs m_top)
# ------------------------------------------------------------------------------
hep.style.use("CMS")

outdir = "results/top_mass"
os.makedirs(outdir, exist_ok=True)

# Define axes for the reconstructed W and top masses
w_mass_axis = hist.axis.Regular(20, 50, 110, name="m_W", label="Reconstructed W Mass [GeV]")
top_mass_axis = hist.axis.Regular(40, 130, 210, name="m_top", label="Reconstructed Top Mass [GeV]")

def get_hist(axes):
    return hist.Hist(*axes, storage=hist.storage.Weight())

# Physics parameters
m_top_nom = 172.5
m_w_nom = 80.4
dm_top = 1.0 # 1 GeV variation for the top mass shift parameter

# JES parameter
# We define a relative JES uncertainty. e.g. JES = 1.0 means nominal
jes_nom = 1.0
djes = 0.02 # 2% variation for the up/dn templates

np.random.seed(42)
n_sig = 50000
n_bkg = 20000

# Background (uncorrelated 2D distribution)
bkg_w_mass = np.random.uniform(50, 110, n_bkg)
bkg_top_mass = np.random.uniform(130, 210, n_bkg)
h_bkg = get_hist((w_mass_axis, top_mass_axis))
h_bkg.fill(bkg_w_mass, bkg_top_mass)

# Function to generate signal given a top mass and a JES factor
def generate_signal(m_top, jes):
    # Base physics values (before JES)
    w_res, top_res = 8.0, 15.0 # resolutions

    # Generate true-like values, then scale by JES to get reconstructed values
    # The true W mass is always m_w_nom (it's a known constant of nature)
    # The true top mass depends on the hypothesis m_top
    w_mass_base = np.random.normal(m_w_nom, w_res, n_sig)
    top_mass_base = np.random.normal(m_top, top_res, n_sig)

    # Apply Jet Energy Scale (JES). This affects both reconstructed masses linearly.
    w_mass_reco = w_mass_base * jes
    top_mass_reco = top_mass_base * jes

    h = get_hist((w_mass_axis, top_mass_axis))
    h.fill(w_mass_reco, top_mass_reco)
    return h

# Nominal templates (m_t = 172.5, JES = 1.0)
h_sig_nom = generate_signal(m_top_nom, jes_nom)

# Top mass variations (JES = 1.0)
h_sig_mtop_up = generate_signal(m_top_nom + dm_top, jes_nom)
h_sig_mtop_dn = generate_signal(m_top_nom - dm_top, jes_nom)

# JES variations (m_t = 172.5, JES shifted by 2%)
h_sig_jes_up = generate_signal(m_top_nom, jes_nom + djes)
h_sig_jes_dn = generate_signal(m_top_nom, jes_nom - djes)

# Background JES variations
# For simplicity, we just shift the background linearly, though in reality
# one would re-evaluate the background estimation with shifted JES.
bkg_w_mass_up, bkg_top_mass_up = bkg_w_mass * (1+djes), bkg_top_mass * (1+djes)
bkg_w_mass_dn, bkg_top_mass_dn = bkg_w_mass * (1-djes), bkg_top_mass * (1-djes)

h_bkg_jes_up = get_hist((w_mass_axis, top_mass_axis))
h_bkg_jes_dn = get_hist((w_mass_axis, top_mass_axis))
h_bkg_jes_up.fill(bkg_w_mass_up, bkg_top_mass_up)
h_bkg_jes_dn.fill(bkg_w_mass_dn, bkg_top_mass_dn)


# Pseudodata
# We inject a true top mass of 173.2 GeV, and a true JES of 1.01 (+1%)
m_top_true = 173.2
jes_true = 1.01
h_sig_true = generate_signal(m_top_true, jes_true)
h_bkg_true = get_hist((w_mass_axis, top_mass_axis))
h_bkg_true.fill(bkg_w_mass * jes_true, bkg_top_mass * jes_true)

h_data = get_hist((w_mass_axis, top_mass_axis))
h_data.view()[...] = h_sig_true.view() + h_bkg_true.view()

# Plot the 1D projections of the nominal templates
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# W mass projection
hep.histplot([h_bkg.project("m_W"), h_sig_nom.project("m_W")], stack=True, histtype='fill', label=['Background', 'Signal'], ax=ax1)
hep.histplot(h_data.project("m_W"), histtype='errorbar', color='black', label='Pseudodata', ax=ax1)
ax1.set_xlabel("Reconstructed W Mass [GeV]")
ax1.set_ylabel("Events")
ax1.legend()

# Top mass projection
hep.histplot([h_bkg.project("m_top"), h_sig_nom.project("m_top")], stack=True, histtype='fill', label=['Background', 'Signal'], ax=ax2)
hep.histplot(h_data.project("m_top"), histtype='errorbar', color='black', label='Pseudodata', ax=ax2)
ax2.set_xlabel("Reconstructed Top Mass [GeV]")
ax2.set_ylabel("Events")
ax2.legend()

plt.savefig(f"{outdir}/templates_2d_projections.png", bbox_inches="tight")

# ------------------------------------------------------------------------------
# Writing the tensor
# ------------------------------------------------------------------------------
writer = tensorwriter.TensorWriter()

# The channel is now a 2D histogram
writer.add_channel(h_data.axes, "mass_region_2d")
writer.add_data(h_data, "mass_region_2d")

writer.add_process(h_sig_nom, "signal", "mass_region_2d", signal=True)
writer.add_process(h_bkg, "background", "mass_region_2d", signal=False)

# Normalization uncertainty for background
writer.add_norm_systematic("bkg_norm", "background", "mass_region_2d", 1.10)

# Unconstrained shape systematic for top mass extraction (affects signal only)
# +1 sigma corresponds to +1 GeV top mass shift
writer.add_systematic(
    [h_sig_mtop_up, h_sig_mtop_dn],
    "top_mass_shift",
    "signal",
    "mass_region_2d",
    symmetrize="average",
    constrained=False,
    noi=True
)

# Constrained shape systematic for JES (affects both signal and background)
# +1 sigma corresponds to +2% JES shift
writer.add_systematic(
    [h_sig_jes_up, h_sig_jes_dn],
    "jes",
    "signal",
    "mass_region_2d",
    symmetrize="average",
    constrained=True
)

writer.add_systematic(
    [h_bkg_jes_up, h_bkg_jes_dn],
    "jes",
    "background",
    "mass_region_2d",
    symmetrize="average",
    constrained=True
)

out_hdf5 = f"{outdir}/top_mass_jes_tensor.hdf5"
writer.write(outfolder=outdir, outfilename="top_mass_jes_tensor")
print(f"Tensor written to {out_hdf5}")

# ------------------------------------------------------------------------------
# Running the fit
# ------------------------------------------------------------------------------
print("\nRunning the fit...")
subprocess.run(
    f"rabbit_fit.py {outdir}/top_mass_jes_tensor.hdf5 -t 0 -o {outdir} --scan top_mass_shift --unblind --postfix mass_fit_jes",
    shell=True,
    check=True
)

# ------------------------------------------------------------------------------
# Interpreting the result
# ------------------------------------------------------------------------------
print("\nPrinting pulls and constraints...")
subprocess.run(
    f"rabbit_print_pulls_and_constraints.py {outdir}/fitresults_mass_fit_jes.hdf5",
    shell=True,
    check=True
)

# ------------------------------------------------------------------------------
# Plotting the likelihood scan
# ------------------------------------------------------------------------------
print("\nPlotting likelihood scan...")
subprocess.run(
    f"rabbit_plot_likelihood_scan.py {outdir}/fitresults_mass_fit_jes.hdf5 --param top_mass_shift -o {outdir}",
    shell=True,
    check=True
)

print(f"\nScan plot saved to: {outdir}/nll_scan_top_mass_shift_data_unblinded.png")