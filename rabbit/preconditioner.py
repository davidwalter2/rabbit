"""Optional preconditioning of the fit parameters.

WHY. The trust-region minimizers solve their subproblem in a *spherical* trust
region, and the Krylov/CG inner solve converges in a number of Hessian-vector
products that grows like sqrt(kappa) of the Hessian. When a block of
parameters is strongly correlated -- typically many unconstrained,
weakly-identified coefficients of a smooth parameterisation, where the basis is
not orthogonal under the data's own weight -- kappa is huge, the inner solve
struggles, and the outer step is rejected. The symptom is outer iterations that
cost minutes and return a bit-identical loss.

WHAT. We reparameterise. With theta the physical parameters, y the internal
ones, theta_ref the point the transform was built at, and a reference Hessian
H0 = L L^T restricted to a selected block:

    theta = theta_ref + T y,     T = L^-T   ->   T^T H0 T = I

so the Hessian in y is the identity at theta_ref and the spherical trust region
in y is an H0-aligned ellipsoid in theta. That is exactly preconditioning, but
obtained as a change of variables, which means **the minimizer is untouched**:
scipy's trust-krylov (GLTR/_trlib) accepts no user preconditioner, and it does
not need to.

The chain rule gives what the scipy callbacks must return:

    loss_y(y)      = loss(theta(y))
    grad_y         = T^T grad_theta
    (H_y p)        = T^T H_theta (T p)

T and T^T are applied as dense matvecs against a cached L^-1, which is formed
once at construction. A triangular solve would do the same flops but is
inherently sequential and so cannot use more than one core; the matvec is a
parallel GEMV and measured ~20x faster at m=2112 (0.09 ms vs 1.9 ms). The
triangular solve is kept as a fallback. Forming the inverse is safe here:
against a direct solve it agrees to ~1e-15 even at cond(H)=1e12.

That is worth stating explicitly because the older *offline* in-situ basis had
to row-normalise its L^-1, which looks like a warning against explicit
inverses. It was not one. There the transform reshaped the basis in which the
histmaker precomputed its delta=0.01 finite variations, so a large T moved the
linearisation point far from where those templates were valid -- a modelling
constraint, not a numerical-stability one. Here T is applied to exact vectors
inside the fit, so that failure mode does not arise.

Parameters outside the selected block are passed through untouched, so the
transform is a no-op there and the block can be as small as one wants.

SCOPE. Preconditioning buys little for constrained nuisances: their unit
Gaussian prior contributes the identity, H = I + J^T W J, so kappa is bounded.
The win is for *unconstrained* parameters and POIs, which is the default scope.

Everything here operates on numpy arrays at the scipy boundary; the fitter
keeps holding physical parameters in its tf.Variable, so nothing downstream
(covariance, impacts, pulls) needs to know that preconditioning happened.
"""

import numpy as np
import scipy.linalg
from wums import logging

logger = logging.child_logger(__name__)

# Sources for the reference matrix. Only "hessian" is implemented so far; the
# others are accepted by the CLI layer as they land.
PRECONDITION_SOURCES = ("hessian",)


class Preconditioner:
    """Affine reparameterisation theta = theta_ref + T y with T = L^-T.

    Use :meth:`identity` for the disabled case: it is an exact no-op, so the
    fitter has a single code path whether or not preconditioning is on.
    """

    def __init__(self, theta_ref, idx=None, chol=None):
        # Correlation condition number of the block before/after the transform,
        # filled in by from_hessian. Kept for diagnostics and tests.
        self.cond_before = None
        self.cond_after = None
        self.theta_ref = np.asarray(theta_ref, dtype=np.float64)
        self.n = self.theta_ref.size
        # idx None  <=> identity transform
        self.idx = None if idx is None else np.asarray(idx, dtype=np.int64)
        self.chol = None if chol is None else np.asarray(chol, dtype=np.float64)
        # Explicit L^-1, formed once so the per-call transform is a dense matvec
        # (parallel GEMV) instead of a triangular solve (inherently sequential).
        # Measured at m=2112: 0.09 ms vs 1.9 ms per application. Falls back to
        # the solve if the inverse cannot be formed.
        self._linv = None
        if (self.idx is None) != (self.chol is None):
            raise ValueError("idx and chol must be given together")
        if self.chol is not None and self.chol.shape != (self.idx.size,) * 2:
            raise ValueError(
                f"chol shape {self.chol.shape} does not match block size {self.idx.size}"
            )
        if self.chol is not None:
            try:
                self._linv = scipy.linalg.solve_triangular(
                    self.chol,
                    np.eye(self.chol.shape[0]),
                    lower=True,
                    trans="N",
                )
            except (scipy.linalg.LinAlgError, ValueError) as ex:
                logger.warning(
                    f"Could not form the explicit inverse ({ex}); "
                    "falling back to triangular solves."
                )
                self._linv = None

    # -- construction ----------------------------------------------------

    @classmethod
    def identity(cls, theta_ref):
        """Disabled preconditioner: y == theta - theta_ref, no block."""
        return cls(theta_ref)

    @property
    def enabled(self):
        return self.idx is not None

    @property
    def nblock(self):
        return 0 if self.idx is None else int(self.idx.size)

    @classmethod
    def from_hessian(cls, hess, theta_ref, idx, ridge=1e-8, max_tries=4):
        """Build from a reference Hessian, restricted to ``idx``.

        ``hess`` is the full [npar, npar] Hessian at ``theta_ref``; only the
        ``idx`` sub-block is used. The block is symmetrised, then a ridge
        proportional to the largest diagonal entry is added until the Cholesky
        succeeds. A block that cannot be factorised at all falls back to the
        identity with a warning: a preconditioner must never break a fit.
        """
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            logger.warning(
                "Preconditioning requested but the selected block is empty; "
                "running unpreconditioned."
            )
            return cls.identity(theta_ref)

        block = np.asarray(hess, dtype=np.float64)[np.ix_(idx, idx)]
        # symmetrise: the autodiff Hessian is symmetric only up to roundoff
        block = 0.5 * (block + block.T)

        diag = np.diag(block)
        scale = float(np.max(diag)) if diag.size else 0.0
        if not np.isfinite(scale) or scale <= 0.0:
            logger.warning(
                "Preconditioning block has no positive diagonal "
                f"(max diag = {scale}); running unpreconditioned."
            )
            return cls.identity(theta_ref)

        cond_before = _cond_corr(block)
        eps = ridge
        for itry in range(max_tries):
            trial = block.copy()
            if eps > 0.0:
                trial[np.diag_indices_from(trial)] += eps * scale
            try:
                chol = scipy.linalg.cholesky(trial, lower=True)
            except scipy.linalg.LinAlgError:
                eps = max(eps, 1e-12) * 100.0
                logger.debug(
                    f"Preconditioner Cholesky failed (try {itry + 1}), "
                    f"raising ridge to {eps:.3g}"
                )
                continue
            # Conditioning actually achieved: L^-1 B L^-T for the *un-ridged*
            # block B. Using the ridged matrix here would return 1 by
            # construction and measure nothing.
            tb = scipy.linalg.solve_triangular(chol, block, lower=True, trans="N")
            tb = scipy.linalg.solve_triangular(chol, tb.T, lower=True, trans="N").T
            cond_after = _cond_corr(tb)
            logger.info(
                f"Preconditioning {idx.size} parameters from the reference Hessian "
                f"(ridge {eps:.3g} x max|diag|): correlation condition number "
                f"{cond_before:.3g} -> {cond_after:.3g} at the reference point"
            )
            out = cls(theta_ref, idx=idx, chol=chol)
            out.cond_before = cond_before
            out.cond_after = cond_after
            return out

        logger.warning(
            "Preconditioning block is not factorisable even with a ridge of "
            f"{eps:.3g} x max|diag|; running unpreconditioned."
        )
        return cls.identity(theta_ref)

    # -- the transform ---------------------------------------------------

    def _apply_T(self, v):
        """T v, i.e. L^-T on the block and the identity elsewhere."""
        if self.idx is None:
            return np.asarray(v, dtype=np.float64)
        out = np.array(v, dtype=np.float64, copy=True)
        if self._linv is not None:
            out[self.idx] = self._linv.T @ out[self.idx]
        else:
            out[self.idx] = scipy.linalg.solve_triangular(
                self.chol, out[self.idx], lower=True, trans="T"
            )
        return out

    def _apply_TT(self, v):
        """T^T v, i.e. L^-1 on the block and the identity elsewhere."""
        if self.idx is None:
            return np.asarray(v, dtype=np.float64)
        out = np.array(v, dtype=np.float64, copy=True)
        if self._linv is not None:
            out[self.idx] = self._linv @ out[self.idx]
        else:
            out[self.idx] = scipy.linalg.solve_triangular(
                self.chol, out[self.idx], lower=True, trans="N"
            )
        return out

    def to_physical(self, y):
        """theta = theta_ref + T y."""
        return self.theta_ref + self._apply_T(y)

    def from_physical(self, theta):
        """y = T^-1 (theta - theta_ref), i.e. L^T on the block."""
        d = np.asarray(theta, dtype=np.float64) - self.theta_ref
        if self.idx is None:
            return d
        out = np.array(d, dtype=np.float64, copy=True)
        out[self.idx] = self.chol.T @ d[self.idx]
        return out

    def grad_to_internal(self, grad):
        """grad_y = T^T grad_theta."""
        return self._apply_TT(grad)

    def hessp_to_internal(self, p, hvp):
        """H_y p = T^T H_theta (T p); ``hvp`` maps a physical-space vector."""
        return self._apply_TT(hvp(self._apply_T(p)))

    def hess_to_internal(self, hess):
        """H_y = T^T H_theta T, for the dense-Hessian minimizers.

        Done as two matrix operations rather than column by column. Note the
        off-diagonal blocks transform too (one-sided), so a block that
        correlates with the rest of the model is handled correctly.
        """
        if self.idx is None:
            return np.asarray(hess, dtype=np.float64)
        out = np.array(hess, dtype=np.float64, copy=True)
        if self._linv is not None:
            # left: T^T acts on the row index; right: T on the column index
            out[self.idx, :] = self._linv @ out[self.idx, :]
            out[:, self.idx] = out[:, self.idx] @ self._linv.T
            return out
        out[self.idx, :] = scipy.linalg.solve_triangular(
            self.chol, out[self.idx, :], lower=True, trans="N"
        )
        out[:, self.idx] = scipy.linalg.solve_triangular(
            self.chol, out[:, self.idx].T, lower=True, trans="N"
        ).T
        return out

    # -- diagnostics -----------------------------------------------------

    def summary(self):
        if self.idx is None:
            return "preconditioning: disabled"
        return f"preconditioning: enabled on {self.idx.size} of {self.n} parameters"


def _cond_corr(mat):
    """Condition number of the *correlation* matrix of ``mat``.

    Scale-invariant, so it measures genuine degeneracy rather than a mismatch
    of units between parameters -- the quantity that actually governs how hard
    the block is to fit.
    """
    d = np.sqrt(np.abs(np.diag(mat)))
    good = d > 0
    if not np.any(good):
        return np.inf
    m = mat[np.ix_(good, good)] / np.outer(d[good], d[good])
    try:
        sv = np.linalg.svd(m, compute_uv=False)
    except np.linalg.LinAlgError:
        return np.inf
    return float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf


def select_indices(
    parms,
    cw,
    frozen_mask,
    expressions=None,
    match_fn=None,
    groups=None,
    group_idxs=None,
):
    """Indices of the parameters to precondition.

    ``expressions`` may name parameters exactly, be regexes matched against the
    full parameter name (via ``match_fn``, the fitter's existing matcher), or
    name a systematic group. With no expressions the default scope is every
    *unconstrained* parameter (cw == 0), which is where preconditioning helps.

    Frozen parameters are always excluded: a dense transform would otherwise
    mix a frozen parameter back into the fit through the other coordinates.
    """
    parms = np.asarray(parms).astype(str)
    n = parms.size
    frozen_mask = np.asarray(frozen_mask, dtype=bool)

    if expressions:
        sel = np.zeros(n, dtype=bool)
        leftover = []
        # NB explicit None checks: groups/group_idxs arrive as numpy arrays,
        # for which `groups or []` raises on the truth-value test.
        gnames = [] if groups is None else list(groups)
        gidxs = [] if group_idxs is None else list(group_idxs)
        by_group = {
            (k.decode() if isinstance(k, bytes) else str(k)): v
            for k, v in zip(gnames, gidxs)
        }
        for expr in expressions:
            if expr in by_group:
                sel[np.asarray(by_group[expr], dtype=np.int64)] = True
            else:
                leftover.append(expr)
        if leftover:
            if match_fn is None:
                raise ValueError("no matcher available for regex selection")
            names = match_fn(leftover, parms)
            sel |= np.isin(parms, names)
    else:
        sel = np.asarray(cw) == 0.0
        logger.info(
            "No --preconditionParams given; defaulting to the unconstrained "
            "parameters (constraint weight 0)."
        )

    sel &= ~frozen_mask
    return np.where(sel)[0]
