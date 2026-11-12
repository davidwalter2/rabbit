
axis_labels = {
    "time": {"label": r"Sidereal time", "unit": "h"},
    "pt_probe": {"label": r"$p_{\text{T}}$", "unit": "GeV"},
    "mll": {"label": r"$m_{\ell \ell}$", "unit": "GeV"},
    "eta_probe": {"label": r"$\eta$", "unit": ""}
}

process_grouping = {
    "backgrounds": [
        "Diboson", "GG", "QCD", "QG_2L", "QG_Lnu", "Top", "W_minus", "Ztautau"
    ]
}

process_colors = {
    "Zmumu": "blue",
    "backgrounds": "red",
}

process_colors = {
    "stat": "red",
    "linearity": "blue",
    "stability": "orange",
    "prefiring_syst": "purple",
    "prefiring_stat": "grey",
}


nuisance_grouping = {
    "liv_unc": [
        "stat",
        "linearity",
        "stability",
        "prefiring_syst",
        "prefiring_stat",
        "bkg",
    ]
}


process_labels = {
    "Data": "Data",
    "sig": r"$X^{\psi}\to\rho\lambda$",
    "bkg": "Bkg.",
    "bkg_2": "M[0]",
}


# axis_labels = {
#     "a": r"Variable a",
#     "b": {"label": r"Variable b", "unit": "GeV"},
#     "x": {"label": r"$\mathit{p}_{T}^{\mu+MET}$", "unit": "GeV"},
# }


# # for impact plots, must be in html style
# impact_labels = {
#     "slope_signal": "Signal slope",
#     "slope_signal_ch0": "Slope signal (ch0)",
#     "slope_signal_ch1": "Slope signal (ch1)",
#     "slope_2_signal_ch1": "Slope signal, 2 (ch1)",
#     "slope_background": "Bkg. signal",
#     "slope_quad_signal_ch0SymDiff": "slope <i>μ</i><sup>2</sup> [diff]",
#     "slope_quad_signal_ch0SymAvg": "slope <i>μ</i><sup>2</sup> [avg]",
#     "slope_lin_signal_ch0SymDiff": "slope <i>x</i><sup>1</sup> [avg]",
#     "slope_lin_signal_ch0SymAvg": "slope <i>x</i><sup>1</sup> [diff]",
#     "norm": "Normalization",
#     "bkg_norm": "Bkg. normalization",
#     "bkg_2_norm": "Bkg. (2) normalization",
#     "Total": "All",
#     "stat": "Data stat.",
#     "binByBinStat": "Simulated samples stat.",
#     "slopes": "Slopes",
#     "slopes_signal": "Signal slopes",
#     "slopes_background": "Background slopes",
#     "sig": "Signal",
# }


# def translate_html_to_latex(n):
#     # transform html style formatting into latex style
#     if "</" in n:
#         n = (
#             f"${n}$".replace("<i>", r"\mathit{")
#             .replace("<sub>", "_{")
#             .replace("<sup>", "^{")
#             .replace("</i>", "}")
#             .replace("</sub>", "}")
#             .replace("</sup>", "}")
#             .replace(" ", r"\ ")
#         )
#     return n


# # same as impact labels but in latex format
# systematics_labels = {k: translate_html_to_latex(v) for k, v in impact_labels.items()}

# # predefined set of grouped impacts to be plotted
# nuisance_grouping = {
#     "min": ["Total", "stat", "binByBinStat", "slopes", "norm"],
# }
