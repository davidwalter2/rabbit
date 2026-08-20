import numpy as np
import tensorflow as tf

from rabbit.regularization.regularizer import Regularizer

AUX_NAME = "insitu_efficiency_bound"


class InSituEfficiencyBound(Regularizer):
    """Keep the in-situ efficiency scale factors inside the physical region.

    The in-situ method parameterises the data/MC efficiency ratio of each step
    as ``SF = 1 + P(theta)`` with ``P`` a Chebyshev polynomial in (pt, ut) whose
    coefficients are *unconstrained* nuisances. A failing leg then contributes

        f_fail = (1 - eMC * SF) / (1 - eMC),

    so the implied data efficiency ``u = eMC * SF`` must stay below 1: at ``u >=
    1`` the fail probability is negative and the model is undefined, not merely
    disfavoured. Nothing in the likelihood enforces that, and in the corners of
    the pt/ut window -- where little data constrains the polynomial -- the fit
    does walk outside it.

    This adds a one-sided penalty on ``u`` per efficiency cell:

        P = sum_cells  relu((u - u0) / w)^2

    ``u0`` defaults to the same bound the histmaker caps at, which is above the
    largest MC efficiency after clamping. The penalty is therefore *exactly*
    zero at theta = 0 and everywhere the model is physical -- it never pulls a
    scale factor away from 1 -- while growing as the square of the distance
    outside. ``w`` sets how fast: at the default the worst cell seen so far
    (u = 1.25) contributes ~600 to the NLL while a cell just barely outside
    contributes ~1e-4.

    A squared hinge rather than a log barrier, deliberately. A barrier's
    curvature diverges at the boundary, which would wreck the conditioning that
    the trust-region solver depends on; the hinge is exactly zero below ``u0``
    (no bias at all where the data constrain the SFs) and has bounded curvature
    above it. It discourages rather than forbids leaving the region -- the
    histmaker caps the linearisation point as a backstop.

    Each cell's ``u`` depends only on the coefficients of its own
    (step, eta, charge) block, so the penalty's Hessian contribution is block
    diagonal in exactly the blocks the preconditioner discovers on its own.

    The efficiency grid and the polynomial basis evaluated on it are read from
    the ``insitu_efficiency_bound`` auxiliary bundle written alongside the
    datacard, so they cannot drift out of step with the coefficient definitions.
    """

    def __init__(self, mapping, dtype, indata=None, threshold=0.9999, width=0.01):
        if indata is None:
            raise ValueError(
                "InSituEfficiencyBound needs the input data to read the "
                f"'{AUX_NAME}' auxiliary bundle"
            )
        aux = getattr(indata, "auxiliary", None) or {}
        if AUX_NAME not in aux:
            raise ValueError(
                f"input file has no '{AUX_NAME}' auxiliary bundle; write it "
                "from setupRabbit with --muonInsituEfficiency"
            )
        bundle = aux[AUX_NAME]

        self.mapping = mapping
        self.dtype = dtype
        self.threshold = float(threshold)
        self.width = float(width)

        # (n_cells,) MC efficiency, (n_cells, k) basis values and the labels of
        # the coefficients they multiply. k is the widest block (idip uses only
        # the pt coefficients and is zero padded).
        self.effmc = tf.constant(
            np.asarray(bundle["effmc"], dtype=np.float64), dtype=dtype
        )
        self.basis = np.asarray(bundle["basis"], dtype=np.float64)
        self.coeff_index = np.asarray(bundle["coeff_index"], dtype=np.int64)
        # fit nuisance -> polynomial coefficient. The histmaker folds the
        # variation step into the stored response, so the two differ by it;
        # taking the nuisance for the coefficient evaluates the polynomial
        # orders of magnitude too large and the penalty swamps the likelihood.
        self.coeff_scale = float(
            np.asarray(bundle.get("coeff_scale", [1.0])).ravel()[0]
        )
        self.labels = [
            x.decode() if isinstance(x, bytes) else str(x) for x in bundle["labels"]
        ]
        if self.coeff_index.max() >= len(self.labels):
            raise ValueError(
                f"{AUX_NAME}: coeff_index references coefficient "
                f"{self.coeff_index.max()} but only {len(self.labels)} labels"
            )
        self.param_index = None
        self.basis_tf = None

    def set_expectations(self, initial_params, initial_observables, parms=None):
        """Resolve the coefficient names against the current parameter layout.

        Positions are re-resolved here rather than cached from construction:
        the fitter can swap in a different ParamModel mid-session, which
        reorders the parameter vector.
        """
        found = self.resolve_indices(parms, self.labels, who=type(self).__name__)
        positions = np.array([found[name] for name in self.labels], dtype=np.int64)
        # cell -> position in the fit parameter vector, via the label list.
        # A gather rather than a sparse matmul: the loss is XLA compiled
        # (_XlaMustCompile), and SparseTensorDenseMatMul has no XLA lowering, so
        # a sparse design matrix aborts the minimizer on the first call.
        self.param_index = tf.constant(positions[self.coeff_index], dtype=tf.int32)
        self.basis_tf = tf.constant(self.basis, dtype=self.dtype)

    def _gather_dense_grad(self, params):
        """Gather the per-cell coefficients with a *dense* gradient.

        tf.gather's gradient is an IndexedSlices, and the fitter's parameter
        vector is a concat of the POI, model-nuisance and theta blocks.
        Scattering an IndexedSlices back into a block that happens to be empty
        aborts the XLA compilation of the loss with "Scatter dimension 0 is of
        size zero", so the minimizer dies on its first call. Summing densely up
        front avoids the scatter entirely and touches exactly the same entries.
        """
        index = self.param_index

        @tf.custom_gradient
        def op(p):
            def grad(dy):
                return tf.math.unsorted_segment_sum(
                    tf.reshape(dy, [-1]), tf.reshape(index, [-1]), tf.size(p)
                )

            return tf.gather(p, index), grad

        return op(params)

    def compute_nll_penalty(self, params, observables):
        if self.param_index is None:
            raise RuntimeError(
                "InSituEfficiencyBound.set_expectations() must run before the "
                "penalty is evaluated"
            )
        theta = self._gather_dense_grad(params)  # (n_cells, k)
        pol = tf.reduce_sum(self.basis_tf * theta, axis=-1)  # (n_cells,)
        u = self.effmc * (1.0 + self.coeff_scale * pol)
        excess = tf.nn.relu((u - self.threshold) / self.width)
        return tf.reduce_sum(tf.square(excess))
