"""Post-fit covariance and EDM at a point that may sit on constraint faces.

At an active face the usual post-fit quantities are misleading: the EDM
measures the distance to the UNCONSTRAINED optimum, and H^-1 lets the fit move
through the face. :func:`face_held_covariance` is the KKT algebra, pure numpy
on the gradient, the loss Hessian and the constraint values / Jacobian /
curvature, so it can be tested on hand-made problems.
:func:`report_constrained_postfit` logs the result and writes it to the
output workspace.
"""

import numpy as np
import scipy
from wums import logging

logger = logging.child_logger(__name__)

#: A face whose release lowers the NLL by less than this is "weakly binding":
#: the optimum beyond it is within one sigma, so the interval along it is
#: one-sided and should be quoted from a likelihood scan.
WEAKLY_BINDING_DNLL = 0.5


def _chol_inv(M):
    """Inverse of a symmetric matrix via Cholesky; None if not positive definite."""
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return None
    Li = np.linalg.inv(L)
    return Li.T @ Li


def face_held_covariance(
    grad,
    hess,
    cval,
    jac,
    lower,
    upper,
    labels,
    constraint_hessian,
    floating=None,
    cov_base=None,
    active_tol=1e-5,
):
    """Face-held and released covariance, EDMs and the active-face summary.

    Parameters
    ----------
    grad, hess : [n], [n, n]
        Loss gradient and Hessian at the point (fitter coordinate).
    cval, jac : [m], [m, n]
        Constraint values and Jacobian there (frozen columns zero).
    lower, upper, labels : length m.
    constraint_hessian : callable
        ``v -> [n, n]``: the Hessian of ``sum_i v_i c_i`` at the point, ``v``
        of length m. Called once, with the multipliers of the active rows.
    floating : array of int, optional
        Parameters the fit floats (default: all). The covariance is formed on
        these; the rest of the returned matrices is taken from ``cov_base``.
    cov_base : [n, n], optional
        Values for the frozen block (default: zeros).
    active_tol : float
        A row within this distance of a bound (in its own units), with a
        multiplier of the binding sign, is an active face.

    Returns
    -------
    dict with
      * ``cov_held`` -- the active faces held: the inverse of the Hessian of
        the Lagrangian on the directions along the faces,
            C = Z (Z^T W Z)^-1 Z^T,   W = H - sum_i mu_i d2c_i,
        Z an orthonormal basis of the null space of A (the active rows'
        gradients), mu their multipliers (grad = A^T mu). The local curvature
        of the profile likelihood on the faces; for a nonlinear face the
        constraint-curvature term is required (for a linear face it is zero).
        Only Z^T W Z has to be positive definite: across a nonlinear face H and
        W can be indefinite at a genuine constrained minimum.
      * ``cov_released``, ``edm_released`` -- H^-1 and 1/2 g^T H^-1 g, every
        face free; None when H is not positive definite (the unconstrained
        problem has no local minimum there).
      * ``edm_tangent`` -- 1/2 g^T C g, the distance to the minimum ON the
        faces (C annihilates the face normals).
      * ``hess_lagrangian`` -- W.
      * ``active`` -- per active row: label, value, bound, distance,
        multiplier, ``release_dnll`` (the NLL gained, to second order, by
        releasing it alone with the other faces held; inf if that leaves no
        local minimum) and ``weakly_binding`` (release_dnll below
        WEAKLY_BINDING_DNLL).
      * ``release_dnll_joint`` -- all faces released at once.
    """
    g = np.asarray(grad, dtype=np.float64).reshape(-1)
    H = np.asarray(hess, dtype=np.float64)
    J = np.asarray(jac, dtype=np.float64)
    cval = np.asarray(cval, dtype=np.float64).reshape(-1)
    lb = np.asarray(lower, dtype=np.float64).reshape(-1)
    ub = np.asarray(upper, dtype=np.float64).reshape(-1)
    labels = [str(lab) for lab in labels]
    n = g.shape[0]
    F = np.arange(n) if floating is None else np.asarray(floating, dtype=int)
    base = np.zeros((n, n)) if cov_base is None else np.asarray(cov_base)

    def embed(CF):
        C = np.array(base, dtype=np.float64, copy=True)
        C[np.ix_(F, F)] = CF
        return C

    # active rows: near a bound, a free parameter can move them, and a
    # multiplier of the binding sign (lower: mu > 0, upper: mu < 0; equality
    # rows always). Rows whose multiplier says the loss pulls INTO the allowed
    # region are not binding and are dropped, and the rest refitted.
    live = np.abs(J[:, F]).sum(axis=1) > 0
    near_lo = np.isfinite(lb) & (np.abs(cval - lb) <= active_tol)
    near_up = np.isfinite(ub) & (np.abs(cval - ub) <= active_tol)
    side = np.where(near_lo & near_up, 0, np.where(near_lo, 1, -1))
    active = np.flatnonzero((near_lo | near_up) & live)
    mu = np.zeros(0)
    while len(active):
        mu = np.linalg.lstsq(J[np.ix_(active, F)].T, g[F], rcond=None)[0]
        wrong = ((side[active] == 1) & (mu < 0)) | ((side[active] == -1) & (mu > 0))
        if not wrong.any():
            break
        active = active[~wrong]
    if not len(active):
        mu = np.zeros(0)

    gF = g[F]
    Hi = _chol_inv(H[np.ix_(F, F)])
    out = {
        "cov_released": None if Hi is None else embed(Hi),
        "edm_released": None if Hi is None else float(0.5 * gF @ Hi @ gF),
    }

    if not len(active):
        if Hi is None:
            raise ValueError(
                "face_held_covariance: no constraint is active and the loss "
                "Hessian is not positive definite on the floating parameters: "
                "this is not a minimum."
            )
        out.update(
            cov_held=out["cov_released"],
            edm_tangent=out["edm_released"],
            hess_lagrangian=H,
            active=[],
            release_dnll_joint=0.0,
        )
        return out

    vfull = np.zeros(len(cval))
    vfull[active] = mu
    Hc = np.asarray(constraint_hessian(vfull), dtype=np.float64)
    W = H - Hc
    WF = W[np.ix_(F, F)]
    A = J[np.ix_(active, F)]

    Z = scipy.linalg.null_space(A)
    T = _chol_inv(Z.T @ WF @ Z)
    if T is None:
        raise ValueError(
            "face_held_covariance: the Hessian of the Lagrangian is not positive "
            "definite along the active faces, so this is not a constrained "
            "minimum (or a direction along a face is flat)."
        )
    Chel = Z @ T @ Z.T
    Chel = 0.5 * (Chel + Chel.T)
    edm_tangent = float(0.5 * gF @ Chel @ gF)

    # NLL gained by releasing face k ALONE (the others held): the EDM of the
    # problem on the directions Z plus z_k, the part of a_k orthogonal to the
    # other active normals, minus the EDM on all faces. By a bordered (Schur)
    # update: 1/2 (z.g - u.T r)^2 / s with u = Z^T W z, s = z.W.z - u.T T u,
    # r = Z^T g. s <= 0: no local minimum once that face is released.
    r = Z.T @ gF
    gains = []
    for k in range(len(active)):
        a = A[k]
        others = np.delete(A, k, axis=0)
        if len(others):
            Q = scipy.linalg.orth(others.T)
            a = a - Q @ (Q.T @ a)
        z = a / np.linalg.norm(a)
        u = Z.T @ (WF @ z)
        s = float(z @ WF @ z - u @ T @ u)
        q = float(z @ gF - u @ T @ r)
        gains.append(0.5 * q * q / s if s > 0 else np.inf)
    Wi = _chol_inv(WF)
    gain_joint = np.inf if Wi is None else float(0.5 * gF @ Wi @ gF) - edm_tangent

    bound = np.where(side[active] == -1, ub[active], lb[active])
    out.update(
        cov_held=embed(Chel),
        edm_tangent=edm_tangent,
        hess_lagrangian=W,
        active=[
            {
                "label": labels[i],
                "value": float(cval[i]),
                "bound": float(b),
                "distance": float(cval[i] - b),
                "multiplier": float(m),
                "release_dnll": float(gn),
                "weakly_binding": bool(gn < WEAKLY_BINDING_DNLL),
            }
            for i, b, m, gn in zip(active, bound, mu, gains)
        ],
        release_dnll_joint=float(gain_joint),
    )
    return out


def report_constrained_postfit(
    result,
    ws,
    active_tol,
    global_impacts=False,
    hessian_impacts=False,
):
    """Log a :func:`face_held_covariance` result and write it to ``ws``.

    Writes ``cov_released`` / ``edmval_released`` (when defined) and the
    ``constraints_postfit`` summary; the caller stores ``cov_held`` as the
    fit's ``cov`` and ``edm_tangent`` as its ``edmval``.

    Returns True when the Hessian-based impacts (``--doImpacts``,
    ``--gaussianGlobalImpacts``, requested via ``hessian_impacts``) must be
    skipped because a face is active.
    """
    faces_active = len(result["active"]) > 0
    released = result["edm_released"]
    logger.info(
        f"edmval (on the active faces): {result['edm_tangent']}; released: "
        + (
            "undefined (the loss Hessian is not positive definite across the faces)"
            if released is None
            else f"{released}"
        )
    )
    for face in result["active"]:
        logger.info(
            f"[constraints] active: {face['label']}: value {face['value']:.6g} "
            f"(bound {face['bound']:.6g}), multiplier {face['multiplier']:.4g}, "
            f"releasing it alone lowers the NLL by {face['release_dnll']:.4g}"
            + (
                f" -- WEAKLY binding (< {WEAKLY_BINDING_DNLL}): the interval along "
                "this face is one-sided; quote it from a likelihood scan"
                if face["weakly_binding"]
                else ""
            )
        )
    if not faces_active:
        logger.info("[constraints] no constraint is active at the minimum")

    if result["cov_released"] is not None:
        ws.add_cov_hist(result["cov_released"], hist_name="cov_released")
        ws.add_value(float(released), "edmval_released")
    ws.add_value(
        {
            "active": result["active"],
            "release_dnll_joint": result["release_dnll_joint"],
            "active_tol": float(active_tol),
        },
        "constraints_postfit",
    )

    if faces_active and global_impacts:
        logger.warning(
            "--globalImpacts at an active constraint face: the per-source impacts "
            "use the face-held covariance, but the grouped decomposition's "
            "remainder ('stat') is not defined there and can come out NaN or the "
            "groups can exceed the total in quadrature."
        )
    skip = faces_active and hessian_impacts
    if skip:
        # Skipped, not refused: only after the fit is it known which faces are
        # active, and raising would throw away a converged fit. Everything else
        # is still written.
        #
        # Why: impacts_parms(hess) and gaussian_global_impacts_parms (through
        # _dxdvars) use the UNCONSTRAINED loss Hessian, which lets the fit move
        # through the face, so they would describe the released problem, not
        # the fit that was run.
        #
        # Known fix (not implemented): compute both from the face-held problem,
        # i.e. the Hessian of the Lagrangian W restricted to the directions
        # along the faces, Z^T W Z, as face_held_covariance does for the
        # covariance. For the Gaussian global impacts that is replacing H^-1 by
        # C_held in the response dx/d(data, constraints) = -C_held
        # d(grad)/d(data, constraints); for the traditional (grouped) impacts,
        # each group's conditional variance taken on the face (the same
        # projection inside impacts_parms). Validate against a fit with the
        # active face frozen, as the covariance was.
        logger.warning(
            "Skipping --doImpacts/--gaussianGlobalImpacts: they are computed from "
            "the unconstrained Hessian, which is wrong at an active constraint "
            "face. --globalImpacts uses the face-held covariance."
        )
    return skip
