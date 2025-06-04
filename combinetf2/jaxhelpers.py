import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import sparse


def simple_sparse_slice0end(in_sparse, end):
    """
    Slice a sparse matrix along axis 0 from 0 to 'end'.
    Assumes in_sparse is a JAX sparse matrix (BCOO format).
    """
    indices = in_sparse.indices
    values = in_sparse.values
    shape = in_sparse.shape

    # Filter rows: select entries where indices[:, 0] < end
    mask = indices[:, 0] < end
    selected_indices = indices[mask]
    selected_values = values[mask]

    # Compute output shape after slicing
    out_shape = (end,) + shape[1:]

    # Create new sparse matrix
    return sparse.BCOO((selected_values, selected_indices), shape=out_shape)


def is_diag(x):
    """Check if matrix is diagonal"""
    return jnp.count_nonzero(x) == jnp.count_nonzero(jnp.diag(jnp.diag(x)))


def jax_edmval_cov(grad, hess):
    """JAX implementation of EDM value and covariance calculation"""
    # Use Cholesky decomposition to detect non-positive-definite case
    try:
        chol = jnp.linalg.cholesky(hess)
    except Exception:
        raise ValueError(
            "Cholesky decomposition failed, Hessian is not positive-definite"
        )

    # Check for NaN values
    if jnp.any(jnp.isnan(chol)):
        raise ValueError(
            "Cholesky decomposition failed, Hessian is not positive-definite"
        )

    gradv = grad[..., None]
    edmval = 0.5 * jnp.dot(gradv.T, jax.scipy.linalg.cho_solve((chol, True), gradv))
    edmval = float(edmval[0, 0])

    cov = jax.scipy.linalg.cho_solve((chol, True), jnp.eye(chol.shape[0]))

    return edmval, cov


def edmval_cov(grad, hess):
    """Calculate EDM value and covariance, choosing backend based on device"""
    return jax_edmval_cov(grad, hess)


def edmval(grad, hess):
    """JAX implementation of EDM value calculation"""
    # Ensure proper shapes
    grad = grad.reshape(-1, 1)  # shape (n, 1)

    # Solve H x = g for x
    x = jnp.linalg.solve(hess, grad)  # shape (n, 1)

    # Compute EDM = 0.5 * g^T x
    edm = 0.5 * jnp.squeeze(jnp.dot(grad.T, x))

    return float(edm)


def cond_number(hess):
    """Calculate condition number of Hessian matrix"""
    return jnp.linalg.cond(hess)


def segment_sum_along_axis(x, segment_ids, idx, num_segments):
    """
    Perform segment sum along a specified axis.
    JAX doesn't have a direct equivalent to tf.math.segment_sum,
    so we implement it using lax.segment_sum.
    """
    # Move the target axis to the front
    perm = [idx] + [i for i in range(len(x.shape)) if i != idx]
    x_transposed = jnp.transpose(x, perm)

    # Apply segment_sum along axis 0
    rebinned = lax.segment_sum(x_transposed, segment_ids, num_segments)

    # Undo the transposition
    reverse_perm = [perm.index(i) for i in range(len(perm))]
    rebinned = jnp.transpose(rebinned, reverse_perm)

    return rebinned
