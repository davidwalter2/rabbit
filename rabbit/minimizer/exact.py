import tensorflow as tf
import numpy as np
import os

from .base import _minimize_trust_region, BaseQuadraticSubproblem

custom_op_path = "/work/submit/david_w/combinetf2-benchmark/custom-op/bazel-bin/tensorflow_zero_out/python/ops/_my_cholesky_op.so"
my_cholesky = tf.load_op_library(custom_op_path)

def _minimize_trust_exact(fun, x0, **trust_region_options):
    """Minimization of scalar function of one or more variables using
    a nearly exact trust-region algorithm.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    initial_tr_radius : float
        Initial trust-region radius.
    max_tr_radius : float
        Maximum value of the trust-region radius. No steps that are longer
        than this value will be proposed.
    eta : float
        Trust region related acceptance stringency for proposed steps.
    gtol : float
        Gradient norm must be less than ``gtol`` before successful
        termination.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.

    """
    return _minimize_trust_region(fun, x0,
                                  subproblem=IterativeSubproblem,
                                  **trust_region_options)

def solve_triangular(matrix, b, lower=True, transpose=False):
    b = tf.expand_dims(b, axis=-1)  # shape [n, 1]
    x = tf.linalg.triangular_solve(matrix, b, lower=lower, adjoint=transpose)
    return tf.squeeze(x, axis=-1)

def solve_cholesky(L, b, lower=False):
    # L is the Cholesky factor; use tf.linalg.cholesky_solve
    b = tf.expand_dims(b, axis=-1)
    x = tf.linalg.cholesky_solve(L, b)  # assumes L @ L^T = A
    return tf.squeeze(x, axis=-1)

def gershgorin_bounds(H):
    """
    Given a square matrix ``H`` compute upper
    and lower bounds for its eigenvalues (Gregoshgorin Bounds).
    Defined ref. [1].

    References
    ----------
    .. [1] Conn, A. R., Gould, N. I., & Toint, P. L.
           Trust region methods. 2000. Siam. pp. 19.
    """
    H_diag = tf.linalg.diag_part(H)
    H_diag_abs = tf.abs(H_diag)
    row_sums = tf.reduce_sum(tf.abs(H), axis=1)
    lb = tf.reduce_min(H_diag + H_diag_abs - row_sums)
    ub = tf.reduce_max(H_diag - H_diag_abs + row_sums)
    return lb, ub

def estimate_smallest_singular_value(L):
    """Given lower triangular matrix ``L`` estimate the smallest singular
    value and the correspondent right singular vector in O(n**2) operations.

    Parameters
    ----------
    L : ndarray
        Square lower triangular matrix.

    Returns
    -------
    s_min : float
        Estimated smallest singular value of the provided matrix.
    z_min : ndarray
        Estimated right singular vector.

    Notes
    -----
    The procedure is based on [1]_ and is done in two steps. First, it finds
    a vector ``e`` with components selected from {+1, -1} such that the
    solution ``w`` from the system ``L.T w = e`` is as large as possible.
    Next it estimates ``L v = w``. The smallest singular value is close
    to ``norm(w)/norm(v)`` and the right singular vector is close
    to ``v/norm(v)``.

    The estimation will be better the more ill-conditioned the matrix is.

    References
    ----------
    .. [1] Cline, A. K., Moler, C. B., Stewart, G. W., Wilkinson, J. H.
           An estimate for the condition number of a matrix.  1979.
           SIAM Journal on Numerical Analysis, 16(2), 368-375.
    """
    L = tf.convert_to_tensor(L)
    n = L.shape[0]
    LT = tf.transpose(L)
    p = tf.zeros([n], dtype=L.dtype)
    w = tf.TensorArray(L.dtype, size=n)

    def body(k, p, w_arr):
        lkk = LT[k, k]
        wp = (1 - p[k]) / lkk
        wm = (-1 - p[k]) / lkk
        pp = p[k+1:] + LT[k+1:, k] * wp
        pm = p[k+1:] + LT[k+1:, k] * wm
        cond = tf.abs(wp) + tf.norm(pp, ord=1) >= tf.abs(wm) + tf.norm(pm, ord=1)
        w_val = tf.where(cond, wp, wm)
        p_next = tf.where(cond, pp, pm)
        p = tf.concat([p[:k+1], p_next], axis=0)
        w_arr = w_arr.write(k, w_val)
        return k+1, p, w_arr

    k = tf.constant(0)
    cond_fn = lambda k, p, w: k < n
    k, p, w_arr = tf.while_loop(cond_fn, body, (k, p, w))
    w = w_arr.stack()

    # Solve L v = w
    v = tf.linalg.triangular_solve(L, tf.expand_dims(w, -1), lower=True)
    v = tf.squeeze(v, -1)
    s_min = tf.norm(w) / tf.norm(v)
    z_min = v / tf.norm(v)
    return s_min, z_min


def singular_leading_submatrix(A, L, k):
    """
    Compute term that makes the leading ``k`` by ``k``
    submatrix from ``A`` singular.

    Parameters
    ----------
    A : ndarray
        Symmetric matrix that is not positive definite.
    L : ndarray
        Lower triangular matrix resulting of an incomplete
        Cholesky decomposition of matrix ``A``.
    k : int
        Positive integer such that the leading k by k submatrix from
        `A` is the first non-positive definite leading submatrix.

    Returns
    -------
    delta : float
        Amount that should be added to the element (k, k) of the
        leading k by k submatrix of ``A`` to make it singular.
    v : ndarray
        A vector such that ``v.T B v = 0``. Where B is the matrix A after
        ``delta`` is added to its element (k, k).
    """

    # this is almost what is done in scipy but with k shifted (see https://github.com/scipy/scipy/blob/main/scipy/optimize/_trustregion_exact.py#L143)

    A_k = A[:k, :k]
    l = L[k-1, :k-1]

    # Compute delta
    delta = tf.tensordot(l, l, axes=1) - A_k[-1,-1]

    # Initialize v
    v = tf.zeros_like(A[:, 0])
    v = tf.tensor_scatter_nd_update(v, [[k-1]], [1.0])

    # Compute the remaining values of v by solving a triangular system.
    if k != 1:
        lower_tri = L[:k-1, :k-1]
        rhs = -l
        v_part = solve_triangular(lower_tri, rhs, lower=True)
        v = tf.tensor_scatter_nd_update(v, [[i] for i in range(k-1)], tf.unstack(v_part))

    return delta, v


# def incomplete_cholesky(A):
#     """
#     Mimics Cholesky decomposition that stops at the first non-PD minor.
#     Returns the largest k such that A[:k, :k] is positive definite.
#     """
#     n = A.shape[0]
#     for k in range(1, n + 1):
#         try:
#             L = tf.linalg.cholesky(A[:k, :k])
#             success = not tf.reduce_any(tf.math.is_nan(L))
#         except tf.errors.InvalidArgumentError as e:
#             success = False
#         if not success:
#             return k - 1
        
#     return n  # Full success

# def incomplete_cholesky(A):
#     """
#     Efficiently find the largest k such that A[:k, :k] is positive definite,
#     using binary search on the leading principal minors.
#     """
#     n = A.shape[0]
#     low = 0
#     high = n

#     while low < high:
#         mid = (low + high + 1) // 2
#         try:
#             L = tf.linalg.cholesky(A[:mid, :mid])
#             success = not tf.reduce_any(tf.math.is_nan(L))
#         except tf.errors.InvalidArgumentError:
#             success = False

#         if success:
#             low = mid  # Try a larger k
#         else:
#             high = mid - 1  # Try a smaller k

#     return low


class IterativeSubproblem(BaseQuadraticSubproblem):
    """Quadratic subproblem solved by nearly exact iterative method.

    Notes
    -----
    This subproblem solver was based on [1]_, [2]_ and [3]_,
    which implement similar algorithms. The algorithm is basically
    that of [1]_ but ideas from [2]_ and [3]_ were also used.

    References
    ----------
    .. [1] A.R. Conn, N.I. Gould, and P.L. Toint, "Trust region methods",
           Siam, pp. 169-200, 2000.
    .. [2] J. Nocedal and  S. Wright, "Numerical optimization",
           Springer Science & Business Media. pp. 83-91, 2006.
    .. [3] J.J. More and D.C. Sorensen, "Computing a trust region step",
           SIAM Journal on Scientific and Statistical Computing, vol. 4(3),
           pp. 553-572, 1983.
    """

    # UPDATE_COEFF appears in reference [1]_
    # in formula 7.3.14 (p. 190) named as "theta".
    # As recommended there it value is fixed in 0.01.
    UPDATE_COEFF = 0.01

    # The subproblem may iterate infinitely for problematic
    # cases (see https://github.com/scipy/scipy/issues/12513).
    # When the `maxiter` setting is None, we need to apply a
    # default. An ad-hoc number (though tested quite extensively)
    # is 25, which is set below. To restore the old behavior (which
    # potentially hangs), this parameter may be changed to zero:
    MAXITER_DEFAULT = 25  # use np.inf for infinite number of iterations

    EPS = tf.experimental.numpy.finfo(tf.float64).eps

    hess_prod = False

    def __init__(self, x, fun, k_easy=0.1, k_hard=0.2, maxiter=None):

        super().__init__(x, fun)

        # When the trust-region shrinks in two consecutive
        # calculations (``tr_radius < previous_tr_radius``)
        # the lower bound ``lambda_lb`` may be reused,
        # facilitating  the convergence. To indicate no
        # previous value is known at first ``previous_tr_radius``
        # is set to -1  and ``lambda_lb`` to None.
        self.previous_tr_radius = -1.0
        self.lambda_lb = None

        # ``k_easy`` and ``k_hard`` are parameters used
        # to determine the stop criteria to the iterative
        # subproblem solver. Take a look at pp. 194-197
        # from reference _[1] for a more detailed description.
        self.k_easy = k_easy
        self.k_hard = k_hard

        # ``maxiter`` optionally limits the number of iterations
        # the solve method may perform. Useful for poorly conditioned
        # problems which may otherwise hang.
        self.maxiter = self.MAXITER_DEFAULT if maxiter is None else maxiter
        if self.maxiter < 0:
            raise ValueError("maxiter must not be set to a negative number"
                             ", use np.inf to mean infinite.")

        self.dimension = self.hess.shape[0]
        self.hess_gersh_lb, self.hess_gersh_ub = gershgorin_bounds(self.hess)
        self.hess_inf = tf.norm(self.hess, ord=np.inf)
        self.hess_fro = tf.norm(self.hess)#, ord='fro')
        self.CLOSE_TO_ZERO = self.dimension * self.EPS * self.hess_inf
            
    def _initial_values(self, tr_radius):
        """Given a trust radius, return a good initial guess for
        the damping factor, the lower bound and the upper bound.
        The values were chosen accordingly to the guidelines on
        section 7.3.8 (p. 192) from [1]_.
        """

        # Upper bound for the damping factor
        hess_norm = tf.minimum(self.hess_fro, self.hess_inf)
        lambda_ub = self.jac_mag / tr_radius + tf.minimum(-self.hess_gersh_lb, hess_norm)
        lambda_ub = tf.nn.relu(lambda_ub) # max(0, lambda_ub)
        # Lower bound for the damping factor
        lambda_lb = self.jac_mag / tr_radius - tf.minimum(self.hess_gersh_ub, hess_norm)
        lambda_lb = tf.maximum(lambda_lb, -tf.reduce_min(tf.linalg.diag_part(self.hess)))
        lambda_lb = tf.nn.relu(lambda_lb) # max(0, lambda_lb)

        # Improve bounds with previous info
        if tr_radius < self.previous_tr_radius:# and self.lambda_lb is not None:
            lambda_lb = tf.maximum(self.lambda_lb, lambda_lb)

        # Initial guess for the damping factor
        lambda_initial = tf.where(
            tf.equal(lambda_lb, 0.0),
            lambda_lb,
            tf.maximum(tf.sqrt(lambda_lb * lambda_ub),
                       lambda_lb + self.UPDATE_COEFF * (lambda_ub - lambda_lb))
        )
        return lambda_initial, lambda_lb, lambda_ub

    def solve(self, tr_radius):
        """Solve quadratic subproblem"""

        lambda_current, lambda_lb, lambda_ub = self._initial_values(tr_radius)
        n = self.dimension
        hits_boundary = True
        already_factorized = False
        niter = 0

        while niter < self.maxiter:
            # print(f"{niter=}")
            # Compute Cholesky factorization
            if already_factorized:
                already_factorized = False
            else:
                H = self.hess + tf.eye(n, dtype=self.hess.dtype) * lambda_current
                L, status = my_cholesky.my_cholesky(H)
                
            niter += 1

            # Check if factorization succeeded
            if status==0 and self.jac_mag > self.CLOSE_TO_ZERO:
                # Successful factorization

                # Solve `L.T L p = s`
                p = solve_cholesky(L, -self.jac)
                p_norm = tf.norm(p)

                # Check for interior convergence
                if p_norm <= tr_radius and lambda_current == 0.0:
                    hits_boundary = False
                    break
                
                # Solve `U.T w = p`
                w = solve_triangular(L, p, transpose=True)
                w_norm = tf.norm(w)

                # Compute Newton step accordingly to
                # formula (4.44) p.87 from ref [2]_.
                delta_lambda = (p_norm / w_norm)**2 * (p_norm - tr_radius) / tr_radius
                lambda_new = lambda_current + delta_lambda
            
                if p_norm < tr_radius:
                    # Inside boundary
                    s_min, z_min = estimate_smallest_singular_value(L)
                    ta, tb = self.get_boundaries_intersections(p, z_min, tr_radius)
                    
                    # Choose `step_len` with the smallest magnitude.
                    # The reason for this choice is explained at
                    # ref [3]_, p. 6 (Immediately before the formula
                    # for `tau`).
                    step_len = tf.cond(tf.abs(ta) < tf.abs(tb), lambda: ta, lambda: tb)

                    # Compute the quadratic term  (p.T*H*p)
                    quadratic_term = tf.tensordot(p, tf.linalg.matvec(self.hess, p), axes=1)

                    # Check stop criteria
                    relative_error = (step_len**2 * s_min**2) / (quadratic_term + lambda_current * tr_radius**2)

                    if relative_error <= self.k_hard:
                        p = p + step_len * z_min
                        break
                    
                    # Update uncertanty bounds
                    lambda_ub = lambda_current
                    lambda_lb = tf.maximum(lambda_lb, lambda_current - s_min**2)

                    # Compute Cholesky factorization
                    L, status = my_cholesky.my_cholesky(H)
                    
                    if status==0:
                        # Update damping factor
                        lambda_current = lambda_new
                        already_factorized = True
                    else:
                        # Update uncertainty bounds
                        lambda_lb = tf.maximum(lambda_lb, lambda_new)
                        # Update damping factor
                        lambda_current = tf.maximum(tf.sqrt(lambda_lb * lambda_ub),
                                                    lambda_lb + self.UPDATE_COEFF * (lambda_ub - lambda_lb))

                else: # Outside boundary
                    # print("Outside boundary")
                    # Check stop criteria
                    relative_error = tf.abs(p_norm - tr_radius) / tr_radius
                    if relative_error <= self.k_easy:
                        break
                    
                    # Update uncertanty bounds
                    lambda_lb = lambda_current

                    # Update damping factor
                    lambda_current = lambda_new

            elif self.jac_mag <= self.CLOSE_TO_ZERO:
                # print("Jac mag close to zero")
                # jac_mag very close to zero

                # Check for interior convergence
                if lambda_current == 0.0:
                    p = tf.zeros([n], dtype=self.jac.dtype)
                    hits_boundary = False
                    break

                s_min, z_min = estimate_smallest_singular_value(L)
                step_len = tr_radius

                # Check stop criteria
                if (step_len**2 * s_min**2) <= self.k_hard * lambda_current * tr_radius**2:
                    p = step_len * z_min
                    break
                
                # Update uncertainty bounds and dampening factor
                lambda_ub = lambda_current
                lambda_lb = tf.maximum(lambda_lb, lambda_current - s_min**2)
                lambda_current = tf.maximum(tf.sqrt(lambda_lb * lambda_ub),
                                            lambda_lb + self.UPDATE_COEFF * (lambda_ub - lambda_lb))

            else: # Unsuccessful factorization
                delta, v = singular_leading_submatrix(H, L, status)
                v_norm = tf.norm(v)

                # Update uncertainty interval
                lambda_lb = tf.maximum(lambda_lb, lambda_current + delta / v_norm**2)

                # Update damping factor
                lambda_current = tf.maximum(
                    tf.sqrt(lambda_lb * lambda_ub),
                    lambda_lb + self.UPDATE_COEFF * (lambda_ub - lambda_lb)
                )

        # print("Subproblem solved")

        self.lambda_lb = lambda_lb
        self.lambda_current = lambda_current
        self.previous_tr_radius = tr_radius
        return p, hits_boundary



