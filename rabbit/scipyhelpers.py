import numpy as np
import scipy


# as scipy.linalg.cho_factor but with the other half of the matrix zeroed for downstream use
def cho_factor_clean(a, lower=False, overwrite_a=False, check_finite=True):
    c = scipy.linalg.cholesky(
        a, lower=lower, overwrite_a=overwrite_a, check_finite=check_finite
    )
    return c, lower


# inverse of a matrix given its Cholesky factorization (output of cho_factor_clean)
# WARNING the provided Cholesky decomposition must have zeros in the other half of the matrix
# which is NOT the case for scipy.linalg.cho_factor
# this avoid the need for the intermediate identity matrix when using scipy.linalg.cho_solve
def cho_inv(chol, overwrite_a=False, check_finite=True):
    c, lower = chol
    c1 = np.asarray_chkfinite(c) if check_finite else np.asarray(c)

    # trtri is the lower level lapack function which computes the inverse of a triangular matrix
    (trtri,) = scipy.linalg.lapack.get_lapack_funcs(("trtri",), (c1,))
    cinv, _ = trtri(c1, lower=lower, overwrite_c=overwrite_a)

    if lower:
        minv = cinv.T @ cinv
    else:
        minv = cinv @ cinv.T

    return minv


def scipy_edmval_cov(grad, hess):
    # FIXME catch this exception to mark failed toys and continue
    try:
        chol = cho_factor_clean(hess, lower=False)
    except scipy.linalg.LinAlgError:
        raise ValueError(
            "Cholesky decomposition failed, Hessian is not positive-definite"
        )

    gradv = grad[..., None]
    edmval = 0.5 * gradv.T @ scipy.linalg.cho_solve(chol, gradv)
    edmval = edmval[0, 0]

    cov = cho_inv(chol, overwrite_a=True)

    return edmval, cov


def scipy_edmval(grad, hess):
    x = np.linalg.solve(hess, grad)
    edmval = 0.5 * np.dot(grad, x)

    return edmval


def scipy_cond_number(hess):
    return np.linalg.cond(hess)
