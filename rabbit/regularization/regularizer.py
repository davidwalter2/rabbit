import numpy as np


class Regularizer:

    def __init__(self, mapping, dtype):
        """
        Initialize the regularization depending on the mapping
        """

    def set_expectations(self, initial_params, initial_observables, parms=None):
        """
        Set the expectations to use in the regularization. Called once per
        parameter layout: the fitter can swap its ParamModel mid-session (the
        saturated goodness-of-fit path wraps it in a CompositeParamModel), which
        reorders and resizes the parameter vector. Do not cache positions across
        calls; re-resolve them by name from ``parms``, the names of every entry
        of ``initial_params``.
        """

    def compute_nll_penalty(self, params, observables):
        """
        Compute the penalty term that gets added to -ln(L), this function should be called in each step of the minimization
        """
        return 0

    @staticmethod
    def resolve_indices(parms, names, who="Regularizer"):
        """Map parameter names to positions in ``parms``, raising if any is absent."""
        if parms is None:
            raise ValueError(
                f"{who}: parameter names are required to resolve positions"
            )
        parms = np.asarray(parms).astype(str)
        index = {name: i for i, name in enumerate(parms)}
        missing = [n for n in names if n not in index]
        if missing:
            raise ValueError(f"{who}: {missing} not in the fit's parameter vector")
        return {n: index[n] for n in names}
