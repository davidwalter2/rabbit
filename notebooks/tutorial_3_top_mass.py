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
# Tutorial 3: Fitting the Top Quark Mass
# ==============================================================================
# In this tutorial we will learn how to extract a parameter of interest like
# the top quark mass from an invariant mass distribution. We'll simulate a
# typical top mass measurement where templates are generated for different
# mass hypotheses.
#
# We use a technique where the mass dependence is implemented as a continuous
# shape systematic variation, mapping the mass variation (e.g., Delta m_t) as
# an unconstrained parameter in the fit.

# ------------------------------------------------------------------------------
# Setting up the environment
# ------------------------------------------------------------------------------
# Emulate RABBIT_BASE (points to the 'rabbit' root)
rabbit_base = str(Path(os.getcwd()).parent.absolute())
os.environ['RABBIT_BASE'] = rabbit_base

# Append rabbit_base to PYTHONPATH so `import rabbit` works
if rabbit_base not in sys.path:
    sys.path.append(rabbit_base)

# Emulate sourcing setup.sh by putting rabbit/bin in PATH
rabbit_bin = os.path.join(rabbit_base, 'bin')
current_path = os.environ.get('PATH', '')
if rabbit_bin not in current_path:
    os.environ['PATH'] = f"{current_path}:{rabbit_bin}"

print("RABBIT_BASE:", os.environ['RABBIT_BASE'])
print("PATH:", os.environ['PATH'])

# ------------------------------------------------------------------------------
# Generating the invariant mass distribution
# ------------------------------------------------------------------------------
hep.style.use("CMS")

# Create an output directory for the results
outdir = "results/top_mass"
os.makedirs(outdir, exist_ok=True)

# Define an axis for the reconstructed top mass
mass_axis = hist.axis.Regular(40, 130, 210, name="m_top", label="Reconstructed Mass [GeV]")

def get_hist(axis):
    return hist.Hist(axis, storage=hist.storage.Weight())

# Nominal top mass and variations
m_top_nom = 172.5
dm = 1.0 # 1 GeV variation

np.random.seed(42)
n_sig = 50000
n_bkg = 20000

# Generate background (exponential or broad distribution)
bkg_mass = np.random.uniform(130, 210, n_bkg)
h_bkg = get_hist(mass_axis)
h_bkg.fill(bkg_mass)

# Function to generate signal given a top mass
def generate_signal(m_top):
    # simple Gaussian resolution
    resolution = 15.0
    mass = np.random.normal(m_top, resolution, n_sig)
    h = get_hist(mass_axis)
    h.fill(mass)
    return h

h_sig_nom = generate_signal(m_top_nom)
h_sig_up = generate_signal(m_top_nom + dm)
h_sig_dn = generate_signal(m_top_nom - dm)

# Pseudodata (we will inject m_top = 173.2 GeV)
m_top_true = 173.2
h_sig_true = generate_signal(m_top_true)
h_data = get_hist(mass_axis)
h_data.view()[...] = h_sig_true.view() + h_bkg.view()

# Plot the templates
fig, ax = plt.subplots(figsize=(8, 6))
hep.histplot([h_bkg, h_sig_nom], stack=True, histtype='fill', label=['Background', 'Signal (m_t=172.5)'], ax=ax)
hep.histplot(h_data, histtype='errorbar', color='black', label=f'Data (m_t={m_top_true})', ax=ax)
ax.legend()
ax.set_ylabel("Events")
plt.savefig(f"{outdir}/templates.png", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------------------
# Writing the tensor
# ------------------------------------------------------------------------------
# We encode the mass dependence as a shape variation (`mass_shift`) and declare
# it as an unconstrained parameter (`constrained=False`). This tells the fitter
# to treat the shift as a freely floating parameter instead of pulling it to a
# Gaussian prior.
writer = tensorwriter.TensorWriter()

writer.add_channel(h_data.axes, "mass_region")
writer.add_data(h_data, "mass_region")

writer.add_process(h_sig_nom, "signal", "mass_region", signal=True)
writer.add_process(h_bkg, "background", "mass_region", signal=False)

# Add standard background normalization uncertainty
writer.add_norm_systematic("bkg_norm", "background", "mass_region", 1.10) # 10% uncertainty

# Add the mass variation as an UNCONSTRAINED systematic
# Since we generated up/dn variations with +/- 1 GeV,
# a value of +1 for this parameter means m_t = 173.5 GeV.
writer.add_systematic(
    [h_sig_up, h_sig_dn],
    "top_mass_shift",
    "signal",
    "mass_region",
    symmetrize="average",
    constrained=False,  # This is the key for a parameter measurement!
    noi=True            # Nuisance parameter of interest
)

out_hdf5 = f"{outdir}/top_mass_tensor.hdf5"
writer.write(outfolder=outdir, outfilename="top_mass_tensor")
print(f"Tensor written to {out_hdf5}")

# ------------------------------------------------------------------------------
# Running the fit
# ------------------------------------------------------------------------------
# Now we run the maximum likelihood fit to extract the top mass. We will scan
# the `top_mass_shift` parameter to compute the likelihood profile and its
# uncertainty. By default, `rabbit_fit.py` scans parameters specified with `--scan`.
print("\nRunning the fit...")
subprocess.run(
    f"rabbit_fit.py {outdir}/top_mass_tensor.hdf5 -t 0 -o {outdir} --scan top_mass_shift --unblind --postfix mass_fit",
    shell=True,
    check=True
)

# ------------------------------------------------------------------------------
# Interpreting the result
# ------------------------------------------------------------------------------
# We can check the parameter pulls and constraints. The `top_mass_shift` is an
# unconstrained parameter, so its "pull" is actually the best fit value of the
# parameter.
print("\nPrinting pulls and constraints...")
subprocess.run(
    f"rabbit_print_pulls_and_constraints.py {outdir}/fitresults_mass_fit.hdf5",
    shell=True,
    check=True
)
# We set up our templates such that +1 on `top_mass_shift` corresponds to +1 GeV
# in mass relative to 172.5 GeV. Our pseudodata was generated with
# m_{top} = 173.2 GeV. The fitted value of `top_mass_shift` should therefore
# be close to 0.7.

# ------------------------------------------------------------------------------
# Plotting the likelihood scan
# ------------------------------------------------------------------------------
# We can plot the likelihood profile we scanned during the fit. The scan output
# provides the 1D Delta ln(L) curve. The minimum of the parabola corresponds
# to the best fit mass shift. The width of the parabola at 2 Delta ln(L) = 1
# gives the 1 sigma uncertainty.
print("\nPlotting likelihood scan...")
subprocess.run(
    f"rabbit_plot_likelihood_scan.py {outdir}/fitresults_mass_fit.hdf5 --param top_mass_shift -o {outdir}",
    shell=True,
    check=True
)

print(f"\nScan plot saved to: {outdir}/nll_scan_top_mass_shift_data_unblinded.png")
