"""
Global impacts from shifting the nuisance parameter of auxiliary (global) observables.
This is the "shifted global observable" method.
Impacts from groups are obtained by adding impacts from individual global observables in quadrature.
"""

import dataclasses

import tensorflow as tf


@dataclasses.dataclass
class GlobalImpactsContext:
    """Bundles the fitter state and configuration needed for all global impact computations."""

    # fit parameters
    x: object
    ubeta: object
    beta_shape: tuple
    # callables
    compute_yields_with_beta_fn: object
    compute_lbeta_fn: object
    compute_lc_fn: object
    # config
    npoi: int
    systgroupidxs: object
    bin_by_bin_stat: bool
    bin_by_bin_stat_mode: str
    global_impacts_from_jvp: bool


def _compute_global_impact_group(d_squared, idxs):
    gathered = tf.gather(d_squared, idxs, axis=-1)
    d_squared_summed = tf.reduce_sum(gathered, axis=-1)
    return tf.sqrt(d_squared_summed)


def _compute_global_impacts_beta0_jvp(ctx, cov_dexpdx, profile=True):
    """
    Computes global impacts from beta parameters via JVP in forward accumulator mode.
    This is fast in case of more beta parameters than explicit parameters (x) and 'cov_dexpdx' has only a few columns.
    It should always be more memory efficient.
    """
    with tf.GradientTape() as t2:
        t2.watch(ctx.ubeta)
        with tf.GradientTape() as t1:
            t1.watch(ctx.ubeta)
            *_, beta = ctx.compute_yields_with_beta_fn(
                profile=profile, compute_norm=False, full=False
            )
            lbeta = ctx.compute_lbeta_fn(beta)
        pdlbetadbeta = t1.gradient(lbeta, ctx.ubeta)

    # pd2lbetadbeta2 is diagonal so we can use gradient instead of jacobian
    pd2lbetadbeta2_diag = t2.gradient(pdlbetadbeta, ctx.ubeta)

    # this is the cholesky decomposition of pd2lbetadbeta2
    sbeta = tf.linalg.LinearOperatorDiag(
        tf.sqrt(tf.reshape(pd2lbetadbeta2_diag, [-1])), is_self_adjoint=True
    )

    impacts_beta_shape = (*ctx.beta_shape, cov_dexpdx.shape[-1])
    impacts_beta0 = tf.zeros(shape=impacts_beta_shape, dtype=cov_dexpdx.dtype)

    if profile:
        # dbeta/dx is None if not profiled (no relation)
        def _tangents(tangent_vector):
            with tf.autodiff.ForwardAccumulator(ctx.x, tangent_vector) as acc:
                *_, beta = ctx.compute_yields_with_beta_fn(
                    profile=True, compute_norm=False, full=False
                )
            return acc.jvp(beta)

        tangents = tf.transpose(cov_dexpdx)
        dbetadx_cov_dexpdx = tf.vectorized_map(_tangents, tangents)

        # flatten all but first axes
        dbetadx_cov_dexpdx = tf.reshape(
            dbetadx_cov_dexpdx, [tf.shape(dbetadx_cov_dexpdx)[0], -1]
        )
        dbetadx_cov_dexpdx = tf.transpose(dbetadx_cov_dexpdx)

        impacts_beta0 += tf.reshape(sbeta @ dbetadx_cov_dexpdx, impacts_beta_shape)

    return impacts_beta0, sbeta


def _compute_global_impacts_beta0(ctx, cov_dexpdx, profile=True):
    """
    Computes global impacts from beta parameters in the traditional mode.
    This is fast in case of less beta parameters than explicit parameters (x) or 'cov_dexpdx' has many columns.
    """
    with tf.GradientTape(persistent=True) as t2:
        t2.watch([ctx.x, ctx.ubeta])
        with tf.GradientTape(persistent=True) as t1:
            t1.watch([ctx.x, ctx.ubeta])
            *_, beta = ctx.compute_yields_with_beta_fn(
                profile=profile, compute_norm=False, full=False
            )
            lbeta = ctx.compute_lbeta_fn(beta)
        pdlbetadbeta = t1.gradient(lbeta, ctx.ubeta)
        dbetadx = t1.jacobian(beta, ctx.x)
    # pd2lbetadbeta2 is diagonal so we can use gradient instead of jacobian
    pd2lbetadbeta2_diag = t2.gradient(pdlbetadbeta, ctx.ubeta)

    # this is the cholesky decomposition of pd2lbetadbeta2
    sbeta = tf.linalg.LinearOperatorDiag(
        tf.sqrt(tf.reshape(pd2lbetadbeta2_diag, [-1])), is_self_adjoint=True
    )

    impacts_beta_shape = (*ctx.beta_shape, cov_dexpdx.shape[-1])
    impacts_beta0 = tf.zeros(shape=impacts_beta_shape, dtype=cov_dexpdx.dtype)

    if profile:
        dbetadx_cov_dexpdx = dbetadx @ cov_dexpdx
        dbetadx_cov_dexpdx = tf.reshape(
            dbetadx_cov_dexpdx, [-1, tf.shape(dbetadx_cov_dexpdx)[-1]]
        )

        impacts_beta0 += tf.reshape(sbeta @ dbetadx_cov_dexpdx, impacts_beta_shape)

    return impacts_beta0, sbeta


def _compute_beta0_impacts(ctx, cov_dexpdx, profile, pdexpdbeta, pd2ldbeta2_pdexpdbeta):
    """Compute beta0 impacts and variance, shared between parms and obs variants."""
    if ctx.global_impacts_from_jvp:
        impacts_beta0, sbeta = _compute_global_impacts_beta0_jvp(
            ctx, cov_dexpdx, profile
        )
    else:
        impacts_beta0, sbeta = _compute_global_impacts_beta0(ctx, cov_dexpdx, profile)

    if pdexpdbeta is not None:
        impacts_beta0 += tf.reshape(sbeta @ pd2ldbeta2_pdexpdbeta, impacts_beta0.shape)

    var_beta0 = tf.reduce_sum(tf.square(impacts_beta0), axis=0)
    impacts_beta0_process = None
    if ctx.bin_by_bin_stat_mode == "full":
        impacts_beta0_process = tf.sqrt(var_beta0)
        var_beta0 = tf.reduce_sum(var_beta0, axis=0)

    return tf.sqrt(var_beta0), impacts_beta0_process, var_beta0


def _compute_global_impacts_x0(ctx, cov_dexpdx):
    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            lc = ctx.compute_lc_fn()
        dlcdx = t1.gradient(lc, ctx.x)
    # d2lcdx2 is diagonal so we can use gradient instead of jacobian
    d2lcdx2_diag = t2.gradient(dlcdx, ctx.x)

    # sc is the cholesky decomposition of d2lcdx2
    sc = tf.linalg.LinearOperatorDiag(tf.sqrt(d2lcdx2_diag), is_self_adjoint=True)
    return sc @ cov_dexpdx


def _compute_grouped_impacts(
    ctx, impacts_theta0_sq, impacts_nobs, impacts_beta0_total, impacts_beta0_process
):
    """Assemble the grouped impacts tensor from all contributions."""
    if ctx.bin_by_bin_stat:
        impacts_grouped = tf.stack([impacts_nobs, impacts_beta0_total], axis=-1)
        if ctx.bin_by_bin_stat_mode == "full":
            impacts_grouped = tf.concat(
                [impacts_grouped, tf.transpose(impacts_beta0_process)], axis=-1
            )
    else:
        impacts_grouped = impacts_nobs[..., None]

    if len(ctx.systgroupidxs):
        impacts_grouped_syst = tf.map_fn(
            lambda idxs: _compute_global_impact_group(impacts_theta0_sq, idxs),
            tf.ragged.constant(ctx.systgroupidxs, dtype=tf.int64),
            fn_output_signature=tf.TensorSpec(
                shape=(impacts_theta0_sq.shape[0],), dtype=impacts_theta0_sq.dtype
            ),
        )
        impacts_grouped_syst = tf.transpose(impacts_grouped_syst)
        impacts_grouped = tf.concat([impacts_grouped_syst, impacts_grouped], axis=-1)

    return impacts_grouped


def global_impacts_parms(ctx, cov, noiidxs):
    # TODO migrate this to a mapping to avoid the below code which is largely duplicated
    idxs_poi = tf.range(ctx.npoi, dtype=tf.int64)
    idxs_noi = tf.constant(ctx.npoi + noiidxs, dtype=tf.int64)
    idxsout = tf.concat([idxs_poi, idxs_noi], axis=0)

    dexpdx = tf.one_hot(idxsout, depth=cov.shape[0], dtype=cov.dtype)
    cov_dexpdx = tf.matmul(cov, dexpdx, transpose_b=True)

    var_total = tf.gather(tf.linalg.diag_part(cov), idxsout)

    impacts_beta0_total, impacts_beta0_process, var_beta0 = None, None, None
    if ctx.bin_by_bin_stat:
        impacts_beta0_total, impacts_beta0_process, var_beta0 = _compute_beta0_impacts(
            ctx, cov_dexpdx, profile=True, pdexpdbeta=None, pd2ldbeta2_pdexpdbeta=None
        )

    impacts_x0 = _compute_global_impacts_x0(ctx, cov_dexpdx)
    impacts_theta0 = tf.transpose(impacts_x0[ctx.npoi :])

    impacts_theta0_sq = tf.square(impacts_theta0)
    var_theta0 = tf.reduce_sum(impacts_theta0_sq, axis=-1)
    var_nobs = var_total - var_theta0
    if ctx.bin_by_bin_stat:
        var_nobs -= var_beta0

    impacts_grouped = _compute_grouped_impacts(
        ctx,
        impacts_theta0_sq,
        tf.sqrt(var_nobs),
        impacts_beta0_total,
        impacts_beta0_process,
    )

    return impacts_theta0, impacts_grouped


def global_impacts_obs(
    ctx,
    cov_dexpdx,
    expvar_flat,
    expvar_shape,
    profile,
    pdexpdbeta=None,
    pd2ldbeta2_pdexpdbeta=None,
    prefit_unconstrained_nuisance_uncertainty=0.0,
):
    """
    Global impacts on observable bins, used inside _expected_with_variance.

    Unlike global_impacts_parms, this variant uses the already-computed cov_dexpdx
    and expvar_flat from the observable Jacobian, and includes the mapping contribution
    via pdexpdbeta / pd2ldbeta2_pdexpdbeta.

    The fully general contribution to the covariance matrix for a factorized likelihood
    L = sum_i L_i can be written as:
        cov_i = dexpdx @ cov_x @ d2L_i/dx2 @ cov_x @ dexpdx.T
    This is totally general and always adds up to the total covariance matrix.

    This can be factorized into impacts only if the individual contributions are rank 1.
    This is not the case in general for the data stat uncertainties, in particular where
    postfit nexpected != nobserved and nexpected is not a linear function of the poi's and
    nuisance parameters x.

    For the systematic and MC stat uncertainties this is equivalent to the more conventional
    global impact calculation (and without needing to insert the uncertainty on the global
    observables "by hand", which can be non-trivial beyond the Gaussian case).
    """
    # protect against inconsistency
    # FIXME this should be handled more generally e.g. through modification of
    # the constraintweights for prefit vs postfit, though special handling of the zero
    # uncertainty case would still be needed
    if (not profile) and prefit_unconstrained_nuisance_uncertainty != 0.0:
        raise NotImplementedError(
            "Global impacts calculation not implemented for prefit case where prefitUnconstrainedNuisanceUncertainty != 0."
        )

    impacts_beta0_total, impacts_beta0_process, var_beta0 = None, None, None
    if ctx.bin_by_bin_stat:
        impacts_beta0_total, impacts_beta0_process, var_beta0 = _compute_beta0_impacts(
            ctx, cov_dexpdx, profile, pdexpdbeta, pd2ldbeta2_pdexpdbeta
        )

    impacts_x0 = _compute_global_impacts_x0(ctx, cov_dexpdx)
    impacts_theta0 = tf.transpose(impacts_x0[ctx.npoi :])

    impacts_theta0_sq = tf.square(impacts_theta0)
    var_theta0 = tf.reduce_sum(impacts_theta0_sq, axis=-1)
    var_nobs = expvar_flat - var_theta0
    if ctx.bin_by_bin_stat:
        var_nobs -= var_beta0

    impacts_grouped = _compute_grouped_impacts(
        ctx,
        impacts_theta0_sq,
        tf.sqrt(var_nobs),
        impacts_beta0_total,
        impacts_beta0_process,
    )

    impacts = tf.reshape(impacts_theta0, [*expvar_shape, impacts_theta0.shape[-1]])
    impacts_grouped = tf.reshape(
        impacts_grouped, [*expvar_shape, impacts_grouped.shape[-1]]
    )

    return impacts, impacts_grouped
