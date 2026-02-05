import numpy as np
import tensorflow as tf
from wums import logging

from rabbit import common
from rabbit.regularization.svd import SVD

logger = logging.child_logger(__name__)

# dictionary with class name and the corresponding filename where it is defined
baseline_regularizations = {
    "SVD": "svd",
}


def load_regularizer(class_name, *args, **kwargs):
    regularization = common.load_class_from_module(
        class_name, baseline_regularizations, base_dir="rabbit.regularization"
    )
    return regularization(*args, **kwargs)


def _compute_curvature(fitter, tau):
    """
    Following Eq.(4.3) from https://iopscience.iop.org/article/10.1088/1748-0221/7/10/T10003/pdf
    """

    # the full derivative d(Li) / d(tau) = pd(Li)/pd(tau) + pd(Li)/pd(x) * d(x)/d(tau)
    #                                    = pd(Li)/pd(tau) - pd(Li)/pd(x) * (pd2(L)/pd(x^2))^-1 * pd2(L)/pd(x)pd(tau)
    # there is no dependency of Li on tau, thus, the first term is 0
    # (pd2(L)/pd(x^2))^-1 is the covariance matrix
    with tf.GradientTape(persistent=True) as t3:
        t3.watch(tau)

        # 1) compute dx/dtau
        with tf.GradientTape(persistent=True) as t2:
            t2.watch(tau)
            with tf.GradientTape() as t1:
                nll = fitter._compute_nll()

            pdLpdx = t1.gradient(nll, fitter.x)

        pd2Lpdx2 = t2.jacobian(pdLpdx, fitter.x)
        pd2Lpdxpdtau = t2.jacobian(pdLpdx, tau)

        chol = tf.linalg.cholesky(pd2Lpdx2)
        dxdtau = -tf.linalg.cholesky_solve(chol, pd2Lpdxpdtau[:, None])
        dxdtau = tf.reshape(dxdtau, [-1])

        # 2) compute pdLx/pdx, pdLy/pdx and pd^2Lx/pdx^2, pd^2Ly/pdx^2
        with tf.GradientTape(persistent=True) as t_inner:
            nexpfullcentral, _, beta = fitter._compute_yields_with_beta(
                profile=False,
                compute_norm=False,
                full=len(fitter.regularizers),
            )

            nexp = nexpfullcentral[: fitter.indata.nbins]

            ln = fitter._compute_ln(nexp)
            lc = fitter._compute_lc()
            lbeta = fitter._compute_lbeta(beta)
            lx = tf.math.log(ln + lc + lbeta)

            x = fitter.get_x()
            penalties = [
                reg.compute_nll_penalty_unweighted(x, nexpfullcentral)
                for reg in fitter.regularizers
            ]
            ly = tf.math.log(tf.add_n(penalties))

        pdLxpdx = t_inner.gradient(lx, fitter.x)
        pdLypdx = t_inner.gradient(ly, fitter.x)

        pdLxpdx_dxdtau = tf.reduce_sum(pdLxpdx * dxdtau)
        pdLypdx_dxdtau = tf.reduce_sum(pdLypdx * dxdtau)

    pdLxpdx_d2xdtau2 = t3.gradient(pdLxpdx_dxdtau, tau)
    pdLypdx_d2xdtau2 = t3.gradient(pdLypdx_dxdtau, tau)

    pd2Lxpdx2 = t3.jacobian(pdLxpdx, fitter.x)
    pd2Lypdx2 = t3.jacobian(pdLypdx, fitter.x)

    dLxdtau = tf.reduce_sum(pdLxpdx * dxdtau)
    dLydtau = tf.reduce_sum(pdLypdx * dxdtau)

    d2Lxdtau2 = (
        tf.reduce_sum(dxdtau * tf.linalg.matvec(pd2Lxpdx2, dxdtau)) + pdLxpdx_d2xdtau2
    )
    d2Lydtau2 = (
        tf.reduce_sum(dxdtau * tf.linalg.matvec(pd2Lypdx2, dxdtau)) + pdLypdx_d2xdtau2
    )

    curvature = (d2Lydtau2 * dLxdtau - d2Lxdtau2 * dLydtau) / tf.pow(
        tf.square(dLxdtau) + tf.square(dLydtau), 1.5
    )

    return curvature


@tf.function
def compute_curvature(fitter, tau):
    return _compute_curvature(fitter, tau)


@tf.function
def neg_curvature_val_grad_hess(fitter, tau):
    with tf.GradientTape() as t2:
        t2.watch(tau)
        with tf.GradientTape() as t1:
            t1.watch(tau)
            val = -1 * _compute_curvature(fitter, tau)
        grad = t1.gradient(val, tau)
    hess = t2.gradient(grad, tau)

    return val, grad, hess


def l_curve_scan_tau(fitter, min=-5, max=5.1, step=0.1):
    tau = SVD.tau
    tau0 = tau.numpy()

    curvatures = []
    tau_steps = np.arange(min, max, step)

    for i, v in enumerate(tau_steps):
        logger.info(f"Iteration {i} with tau = {v}")

        tau.assign(v)
        cb = fitter.minimize()
        val = compute_curvature(fitter, tau).numpy()
        curvatures.append(val)

        logger.info(f"Curvature (value) = {val}")

    # set tau back to the original value
    tau.assign(tau0)

    return tau_steps, np.array(curvatures)


def l_curve_optimize_tau(fitter):
    # find the tau where the curvature is maximum, minimize curvature w.r.t. tau

    tau = SVD.tau

    edm = 1
    i = 0
    while i < 50 and edm > 1e-16:
        cb = fitter.minimize()
        logger.info(f"Iteration {i}")

        val, grad, hess = neg_curvature_val_grad_hess(fitter, tau)

        logger.info(f"Curvature (value) = {-val}")
        logger.info(f"Curvature (gradient) = {grad}")
        logger.info(f"Curvature (hessian) = {hess}")

        # eps = 1e-8
        # safe_hess = tf.where(hess != 0, hess, tf.ones_like(hess))
        # step = grad / (tf.abs(safe_hess) + eps)
        step = grad / hess
        logger.info(f"Curvature (step) = {-step}")
        tau.assign_sub(step)
        edm = tf.reduce_max(0.5 * tf.square(grad) * tf.abs(hess))
        i = i + 1

        logger.debug(f"Curvature edm = {edm}")
        logger.debug(f"Curvature tau = {tau}")

    logger.info(f"Optimization terminated")
    logger.info(f"  edm: {edm}")
    logger.info(f"  maximum curvature: {-val}")
    logger.info(f"  tau: {tau.numpy()}")

    return tau.numpy(), val.numpy()
