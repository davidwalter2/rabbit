"""Tests for the in-situ efficiency physical-region penalty.

The in-situ scale factors are unconstrained nuisances and the likelihood alone
does not stop the implied data efficiency eMC*(1+P) from reaching 1, where the
fail probability turns negative and the model is undefined. These cover the
properties the penalty has to have to be usable: silent inside the region,
growing outside it, and pushing in the direction that restores physicality.
"""

from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from rabbit.regularization.insitu import AUX_NAME, InSituEfficiencyBound


def make_indata(effmc, basis, coeff_index, labels):
    return SimpleNamespace(
        auxiliary={
            AUX_NAME: {
                "effmc": np.asarray(effmc, dtype=float),
                "basis": np.asarray(basis, dtype=float),
                "coeff_index": np.asarray(coeff_index, dtype=int),
                "labels": list(labels),
            }
        }
    )


def make_bound(effmc=(0.9, 0.5), **kw):
    """Two cells, each driven by its own single coefficient with basis 1, so
    u_i = effmc_i * (1 + theta_i), f_fail_i = (1 - u_i)/(1 - effmc_i), and the
    expected penalty is analytic."""
    n = len(effmc)
    indata = make_indata(
        effmc=effmc,
        basis=np.ones((n, 1)),
        coeff_index=np.arange(n).reshape(n, 1),
        labels=[f"c{i}" for i in range(n)],
    )
    reg = InSituEfficiencyBound(None, tf.float64, indata=indata, **kw)
    reg.set_expectations(None, None, parms=[f"c{i}" for i in range(n)])
    return reg


def penalty(reg, theta):
    return float(reg.compute_nll_penalty(tf.constant(theta, dtype=tf.float64), None))


def test_zero_at_the_nominal_point():
    """theta = 0 means SF = 1: the penalty must not pull the SFs anywhere."""
    reg = make_bound(effmc=(0.9999, 0.99, 0.5))
    assert penalty(reg, [0.0, 0.0, 0.0]) == 0.0


def test_zero_everywhere_inside_the_physical_region():
    reg = make_bound(effmc=(0.5,), failFloor=0.05, failWidth=0.02)
    # f_fail = (1 - 0.5(1+theta))/0.5 stays above 0.05 until theta ~ 0.9
    for theta in (-0.5, 0.0, 0.5, 0.8):
        assert penalty(reg, [theta]) == 0.0


def test_grows_quadratically_outside():
    """eMC = 0.5 so f_fail = (1 - 0.5(1+theta))/0.5 = 1 - theta; with a floor of
    0.5 and width 0.1 the shortfall is ((0.5 - (1-theta))/0.1) = (theta-0.5)/0.1."""
    reg = make_bound(effmc=(0.5,), failFloor=0.5, failWidth=0.1)
    for theta, expect in [(0.5, 0.0), (0.6, 1.0), (0.7, 4.0), (0.8, 9.0)]:
        assert np.isclose(penalty(reg, [theta]), expect), theta


def test_penalises_the_unphysical_cell_only():
    """One cell outside must not be masked by others sitting comfortably inside."""
    reg = make_bound(effmc=(0.9, 0.5), failFloor=0.05, failWidth=0.02)
    assert penalty(reg, [0.0, 0.0]) == 0.0
    # cell 0: u = 0.9*1.2 = 1.08, f_fail = (1-1.08)/0.1 = -0.8
    one_outside = penalty(reg, [0.2, 0.0])
    assert np.isclose(one_outside, ((0.05 - (-0.8)) / 0.02) ** 2)


def test_threshold_scales_with_the_cells_own_headroom():
    """The reason for writing the penalty on f_fail rather than on u.

    A high-eMC cell has little room between eMC and 1, so a fixed u threshold
    would penalise it for the MC efficiency alone. A floor on f_fail leaves it
    alone until the scale factor eats into that headroom.
    """
    high = make_bound(effmc=(0.9975,), failFloor=0.05, failWidth=0.02)
    # u = 0.9959, which a fixed u0 = 0.995 would have penalised
    theta = 0.9959 / 0.9975 - 1.0
    assert penalty(high, [theta]) == 0.0
    # the equivalent per-cell bound is u <= 1 - 0.05*(1-eMC) = 0.999875
    bound = 1.0 - 0.05 * (1.0 - 0.9975)
    assert penalty(high, [(bound - 1e-6) / 0.9975 - 1.0]) == 0.0
    assert penalty(high, [(bound + 1e-4) / 0.9975 - 1.0]) > 0.0


def grad_hess(reg, theta_0, with_penalty=True):
    """Gradient and Hessian of the penalty as the fitter takes them: watched
    tensor (not Variable), gradient then jacobian, under XLA.

    The penalty is summed with a dense quadratic standing in for the likelihood
    terms, because that is the only way it is ever differentiated. Alone, the
    gather's gradient is an IndexedSlices; summed with a dense one it
    densifies, so testing it in isolation would test a configuration that never
    occurs. Returns the penalty's own contribution, with the stand-in removed.

    XLA matters here: the loss is compiled with _XlaMustCompile, so an op with
    no XLA lowering -- a sparse matmul, say -- aborts the minimizer on its first
    call rather than merely running slowly.
    """

    @tf.function(jit_compile=True)
    def compute(x):
        with tf.GradientTape() as t2:
            t2.watch(x)
            with tf.GradientTape() as t1:
                t1.watch(x)
                total = tf.reduce_sum(x * x)  # dense stand-in for the NLL
                if with_penalty:
                    total += reg.compute_nll_penalty(x, None)
            g = t1.gradient(total, x)
        return g, t2.jacobian(g, x)

    g, h = compute(tf.constant(theta_0, dtype=tf.float64))
    # subtract the stand-in analytically: d/dx sum(x^2) = 2x, d2/dx2 = 2I
    x = np.asarray(theta_0, dtype=float)
    return g.numpy() - 2.0 * x, h.numpy() - 2.0 * np.eye(x.size)


def test_gradient_pushes_the_scale_factor_back_down():
    """The whole point: the fit must feel a force toward the physical region."""
    reg = make_bound(effmc=(0.9,), failFloor=0.05, failWidth=0.02)
    grad, _ = grad_hess(reg, [0.2])
    # penalty rises with theta, so descending it lowers the scale factor
    assert grad[0] > 0.0
    # and no force at all while inside
    assert grad_hess(reg, [0.0])[0][0] == 0.0


def test_curvature_is_bounded_unlike_a_barrier():
    """A log barrier would diverge at the boundary and wreck the conditioning
    the trust-region solver depends on; the hinge must not."""
    reg = make_bound(effmc=(0.9,), failFloor=0.05, failWidth=0.02)
    curvatures = [float(grad_hess(reg, [t])[1][0, 0]) for t in (0.12, 0.15, 0.2, 0.5)]
    # exactly constant for a quadratic hinge, and finite everywhere
    assert np.allclose(curvatures, curvatures[0])
    assert np.isfinite(curvatures).all()


def test_second_derivative_survives_xla_compilation():
    """The regression that a sparse design matrix introduced: the penalty must
    compile inside the fitter's jitted loss, gradient and Hessian included."""
    reg = make_bound(effmc=(0.9, 0.5), failFloor=0.05, failWidth=0.02)
    grad, hess = grad_hess(reg, [0.2, 0.0])
    assert np.isfinite(grad).all() and np.isfinite(hess).all()
    assert hess.shape == (2, 2)
    # only the cell that is outside contributes curvature
    assert hess[0, 0] > 0.0 and hess[1, 1] == 0.0


def test_resolves_positions_by_name_not_by_construction_order():
    """The fitter can reorder the parameter vector, so positions must be looked
    up from the names it reports at arming time."""
    reg = make_bound(effmc=(0.9, 0.5), failFloor=0.05, failWidth=0.02)
    reg.set_expectations(None, None, parms=["other", "c1", "c0"])
    # c0 now sits at position 2: only a value there may drive the first cell
    assert penalty(reg, [0.2, 0.0, 0.0]) == 0.0
    assert penalty(reg, [0.0, 0.0, 0.2]) > 0.0


def test_missing_bundle_is_an_error_not_a_silent_no_op():
    """A silently inert penalty would look exactly like a fit that stayed
    physical on its own."""
    for aux in ({}, None):
        try:
            InSituEfficiencyBound(
                None, tf.float64, indata=SimpleNamespace(auxiliary=aux)
            )
        except ValueError:
            continue
        raise AssertionError("expected a ValueError for a missing bundle")


def test_padded_coefficients_contribute_nothing():
    """idip blocks are zero padded to the width of the ut-dependent ones."""
    indata = make_indata(
        effmc=[0.9],
        basis=[[1.0, 0.0]],  # second column padded
        coeff_index=[[0, 1]],
        labels=["c0", "c1"],
    )
    reg = InSituEfficiencyBound(
        None, tf.float64, indata=indata, failFloor=0.05, failWidth=0.02
    )
    reg.set_expectations(None, None, parms=["c0", "c1"])
    # moving the padded coefficient must not change anything
    assert penalty(reg, [0.0, 5.0]) == 0.0
    assert np.isclose(penalty(reg, [0.2, 0.0]), penalty(reg, [0.2, 5.0]))


def test_survives_an_empty_block_in_the_parameter_concat():
    """fitter.get_x() concatenates the POI, model-nuisance and theta blocks, and
    a fit with no POIs makes the first one empty. A gather's IndexedSlices
    gradient cannot be scattered into a zero-size block under XLA, which killed
    the minimizer on its first call; the penalty must differentiate densely.
    """
    reg = make_bound(effmc=(0.9, 0.5), failFloor=0.05, failWidth=0.02)

    @tf.function(jit_compile=True)
    def loss_and_grad(empty_block, theta):
        with tf.GradientTape() as tape:
            tape.watch([empty_block, theta])
            x = tf.concat([empty_block, theta], axis=0)
            p = reg.compute_nll_penalty(x, None)
        return p, tape.gradient(p, [empty_block, theta])

    value, (g_empty, g_theta) = loss_and_grad(
        tf.constant(np.zeros(0), dtype=tf.float64),
        tf.constant([0.2, 0.0], dtype=tf.float64),
    )
    assert np.isfinite(float(value))
    assert g_empty.shape == (0,)
    assert g_theta.numpy()[0] > 0.0  # the cell outside the region pushes back


def test_the_bound_is_two_sided():
    """Only the fail side was bounded, so nothing stopped the scale factor
    collapsing towards zero efficiency except the helper's hard throw."""
    reg = make_bound(
        effmc=(0.9,), failFloor=0.05, failWidth=0.02, passFloor=0.5, passWidth=0.1
    )
    assert penalty(reg, [0.0]) == 0.0  # SF = 1, both sides quiet
    assert penalty(reg, [-0.2]) == 0.0  # SF = 0.8, still inside
    # SF = 0.4 is below the 0.5 floor: shortfall (0.5-0.4)/0.1 = 1
    assert np.isclose(penalty(reg, [-0.6]), 1.0)
    # and it pushes the scale factor back up
    grad, _ = grad_hess(reg, [-0.6])
    assert grad[0] < 0.0


def test_the_two_sides_are_independent():
    """A cell pinned against one side must not be penalised by the other."""
    reg = make_bound(
        effmc=(0.9,), failFloor=0.05, failWidth=0.02, passFloor=0.5, passWidth=0.1
    )
    high = penalty(reg, [0.2])  # u = 1.08, fail side only
    low = penalty(reg, [-0.6])  # SF = 0.4, pass side only
    assert high > 0.0 and low > 0.0
    # disabling the pass floor leaves the fail-side penalty untouched
    only_fail = make_bound(effmc=(0.9,), failFloor=0.05, failWidth=0.02, passFloor=0.0)
    assert np.isclose(penalty(only_fail, [0.2]), high)
    assert penalty(only_fail, [-0.6]) == 0.0


def test_smoothing_keeps_the_penalty_zero_inside_the_bound():
    """The smoothing must not leak a penalty where the bound is satisfied --
    that property is why a hinge was chosen over a barrier."""
    reg = make_bound(effmc=(0.9, 0.5), smooth=1.0)
    assert penalty(reg, [0.0, 0.0]) == 0.0
    assert penalty(reg, [-0.3, -0.3]) == 0.0


def test_smoothing_has_continuous_curvature_through_the_kink():
    """relu(x)^2 jumps from curvature 0 to 2 at the kink, so cells crossing it
    change the Hessian discontinuously and the preconditioner goes stale."""
    sharp = make_bound(effmc=(0.9,), smooth=0.0)
    smooth = make_bound(effmc=(0.9,), smooth=1.0)

    # f_fail = (1 - eMC(1+theta))/(1 - eMC) crosses the 0.05 floor here; the
    # samples have to straddle it by much less than `smooth`, since that is the
    # scale over which the smoothing acts
    eff, floor, width = 0.9, 0.05, 0.02
    cross = (1.0 - floor * (1.0 - eff)) / eff - 1.0
    dshortfall_dtheta = eff / (1.0 - eff) / width

    def curvature(reg, shortfall):
        return grad_hess(reg, [cross + shortfall / dshortfall_dtheta])[1][0, 0]

    # just inside the bound both are silent
    assert curvature(sharp, -0.02) == 0.0
    assert curvature(smooth, -0.02) == 0.0

    # just outside, the sharp hinge is already at its full curvature: it jumps
    full = curvature(sharp, 0.04)
    assert np.isclose(curvature(sharp, 0.01), full)

    # the smoothed one instead approaches zero as the kink is approached, and
    # does so linearly (d2/dx2 of x^3/(x+d) is 6x/d for x << d)
    c1, c4 = curvature(smooth, 0.01), curvature(smooth, 0.04)
    assert 0.0 < c1 < c4 < full
    assert np.isclose(c4 / c1, 4.0, rtol=0.1), (c1, c4)


def test_smoothing_agrees_with_the_hinge_far_outside():
    """It must only differ near the kink, not change what the bound means."""
    sharp = make_bound(effmc=(0.9,), smooth=0.0)
    smooth = make_bound(effmc=(0.9,), smooth=0.5)
    deep = 0.5  # far outside: shortfall >> smooth
    a, b = penalty(sharp, [deep]), penalty(smooth, [deep])
    assert abs(b - a) / a < 0.05, (a, b)
