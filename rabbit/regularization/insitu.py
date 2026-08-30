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

    The penalty is written on the fail factor itself, one-sided:

        P = sum_cells  relu((f_min - f_fail) / w)^2

    rather than on ``u`` against a fixed threshold. Those are not equivalent,
    and the difference matters. A fixed ``u0`` penalises a cell whose *MC*
    efficiency alone already exceeds it: at eMC = 0.9975 and u0 = 0.995 the cell
    pays a penalty before the scale factor has done anything, which is a
    systematic pull on the highest-efficiency regions for no physical reason. On
    a real combined fit 670 of the 754 cells above such a threshold were of
    exactly that kind, with |SF - 1| < 0.005.

    A floor on f_fail is equivalent to the per-cell bound

        u <= 1 - f_min * (1 - eMC),

    which scales with each cell's own headroom: the same f_min leaves a cell at
    eMC = 0.9975 alone until u = 0.99988, while acting on one at eMC = 0.83 from
    u = 0.9915. It also says something interpretable -- the scale factors may not
    suppress a fail category below ``f_min`` of its MC prediction -- which is the
    quantity that actually decides whether the fail regions keep enough events to
    constrain anything. Capping ``u`` at a fixed value left fail factors of 7e-4,
    emptying bins that held thousands of data events.

    A second, looser floor bounds the other side: f_pass = SF >= p_min. Together
    they confine the implied data efficiency to a band strictly inside (0, 1),

        p_min * eMC  <=  u  <=  1 - f_min * (1 - eMC),

    with both margins scaling per cell -- the upper one with the headroom to 1,
    the lower one with eMC itself.

    The penalty is exactly zero at theta = 0 for any f_min < 1 and p_min < 1,
    since f_fail = f_pass = 1 there, so it never pulls a scale factor away
    from 1.

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

    # u = eMC * (1 + delta * P(theta)) is built from the fitted coefficients
    # and the MC efficiency grid carried in the aux bundle; the predicted
    # yields never enter. Declaring that makes the penalty available to
    # bins-sharded multi-device fits, where no device holds the full yields.
    needs_observables = False

    def __init__(
        self,
        mapping,
        dtype,
        indata=None,
        failFloor=0.05,
        failWidth=0.02,
        passFloor=0.5,
        passWidth=0.1,
        smooth=0.0,
        smoothKind="rational",
    ):
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
        self.fail_floor = float(failFloor)
        self.fail_width = float(failWidth)
        self.pass_floor = float(passFloor)
        self.pass_width = float(passWidth)
        self.smooth = float(smooth)
        self.smooth_kind = str(smoothKind)

        # (n_cells,) MC efficiency, (n_cells, k) basis values and the labels of
        # the coefficients they multiply. k is the widest block (idip uses only
        # the pt coefficients and is zero padded).
        effmc = np.asarray(bundle["effmc"], dtype=np.float64)
        self.effmc = tf.constant(effmc, dtype=dtype)
        # 1 - eMC, the denominator of the fail factor. The bundle clamps eMC to
        # effMC_max < 1, so this is bounded away from zero.
        self.headroom = tf.constant(1.0 - effmc, dtype=dtype)
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

    def _hinge(self, x):
        """Squared hinge on ``x``, optionally smoothed through the kink.

        relu(x)^2 has a *discontinuous* second derivative: curvature 2 above the
        kink and exactly 0 below it. The preconditioner whitens the Hessian for
        one set of violating cells, and every cell that crosses the kink as the
        fit descends changes that Hessian abruptly -- so the transform stops
        matching the problem, which is an active-set problem being handed to a
        smooth trust-region solver.

        ``smooth`` replaces it with x^3/(x + smooth), which is still exactly zero
        for x <= 0 (no bias where the bound is satisfied), agrees with x^2 once
        x >> smooth, and has continuous value, gradient and curvature at the
        kink -- all three vanish as x -> 0+. Curvature stays bounded, unlike a
        log barrier.
        """
        if self.smooth <= 0.0:
            return tf.square(tf.nn.relu(x))
        if self.smooth_kind == "softplus":
            # the textbook smoothing, but softplus(0) = ln2, so it leaks a
            # penalty of (smooth*ln2)^2 where the bound is satisfied and decays
            # only exponentially below the kink -- the bias the hinge was chosen
            # to avoid, reintroduced at a smaller scale
            return tf.square(self.smooth * tf.math.softplus(x / self.smooth))
        r = tf.nn.relu(x)
        return r * r * r / (r + self.smooth)

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

    def compute_nll_penalty(self, params, observables=None):
        if self.param_index is None:
            raise RuntimeError(
                "InSituEfficiencyBound.set_expectations() must run before the "
                "penalty is evaluated"
            )
        theta = self._gather_dense_grad(params)  # (n_cells, k)
        pol = tf.reduce_sum(self.basis_tf * theta, axis=-1)  # (n_cells,)
        u = self.effmc * (1.0 + self.coeff_scale * pol)
        # f_fail goes negative once u passes 1, so the same expression covers
        # both "too small a fail probability" and "no fail probability at all"
        f_fail = (1.0 - u) / self.headroom
        penalty = tf.reduce_sum(
            self._hinge((self.fail_floor - f_fail) / self.fail_width)
        )

        if self.pass_floor > 0.0:
            # The other side of the physical region. Only f_fail was bounded, so
            # nothing stopped the scale factor collapsing towards zero
            # efficiency except the helper's hard throw at 1 + P <= 0 -- a cliff
            # rather than a bound, which is why the curves drop sharply in
            # places. f_pass = SF is the pass-side counterpart of f_fail.
            #
            # Unlike the fail floor this is a smoothness guard, not a
            # requirement: the pass category is where the statistics are, so
            # nothing is being emptied. Keep it loose enough to catch only
            # pathology.
            f_pass = 1.0 + self.coeff_scale * pol
            penalty += tf.reduce_sum(
                self._hinge((self.pass_floor - f_pass) / self.pass_width)
            )
        return penalty
