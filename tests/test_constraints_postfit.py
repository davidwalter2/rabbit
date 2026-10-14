"""rabbit.constraints.postfit.face_held_covariance on hand-made problems.

Pure numpy, no fitter: the KKT algebra is the subtle part of the constrained
post-fit, so it is pinned on problems whose answer is known in closed form.

The workhorse is f(x, y) = 1/2 h (x - 2)^2 + 1/2 y^2 (+ const) with the
nonlinear constraint c = 1 - x - kappa y^2 >= 0, binding at (1, 0):
  grad f = (-h, 0) = mu grad c with grad c = (-1, 0), so mu = h > 0;
  d2c = diag(0, -2 kappa), W = H - mu d2c = diag(h, 1 + 2 h kappa);
  along the face x = 1 - kappa y^2 the loss is 1/2 h (1 + kappa y^2)^2 +
  1/2 y^2, curvature 1 + 2 h kappa in y -- what C_held must invert;
  releasing the face gains 1/2 h (exactly, the loss is quadratic).
"""

import numpy as np
import pytest

from rabbit.constraints.postfit import face_held_covariance


def _problem(h=1.0, kappa=0.7):
    g = np.array([-h, 0.0])
    H = np.diag([h, 1.0])
    cval = np.array([0.0])
    J = np.array([[-1.0, 0.0]])

    def chess(v):
        return v[0] * np.diag([0.0, -2.0 * kappa])

    return g, H, cval, J, chess


def test_nonlinear_face_closed_form():
    h, kappa = 1.0, 0.7
    g, H, cval, J, chess = _problem(h, kappa)
    r = face_held_covariance(g, H, cval, J, [0.0], [np.inf], ["c"], chess)
    np.testing.assert_allclose(
        r["cov_held"], np.diag([0.0, 1.0 / (1.0 + 2 * h * kappa)]), atol=1e-14
    )
    np.testing.assert_allclose(r["cov_released"], np.linalg.inv(H))
    np.testing.assert_allclose(r["hess_lagrangian"], np.diag([h, 1 + 2 * h * kappa]))
    (face,) = r["active"]
    assert face["multiplier"] == pytest.approx(h)
    assert face["release_dnll"] == pytest.approx(0.5 * h)
    assert r["release_dnll_joint"] == pytest.approx(0.5 * h)
    assert r["edm_tangent"] == pytest.approx(0.0, abs=1e-15)
    assert r["edm_released"] == pytest.approx(0.5 * h)


def test_weakly_binding_flag():
    # release gain 1/2 h: below 0.5 for h < 1, not above
    for h, weak in ((0.6, True), (1.4, False)):
        g, H, cval, J, chess = _problem(h)
        r = face_held_covariance(g, H, cval, J, [0.0], [np.inf], ["c"], chess)
        assert r["active"][0]["weakly_binding"] is weak


def test_indefinite_across_the_face():
    """Bend the loss down across the face: H indefinite, still a constrained
    minimum. The held covariance is unchanged; the released view does not
    exist and releasing the face is unbounded."""
    h, kappa = 1.0, 0.7
    g, H, cval, J, chess = _problem(h, kappa)
    Hbad = np.diag([-3.0, 1.0])
    r = face_held_covariance(g, Hbad, cval, J, [0.0], [np.inf], ["c"], chess)
    np.testing.assert_allclose(
        r["cov_held"], np.diag([0.0, 1.0 / (1.0 + 2 * h * kappa)]), atol=1e-14
    )
    assert r["cov_released"] is None and r["edm_released"] is None
    assert r["active"][0]["release_dnll"] == np.inf
    assert not r["active"][0]["weakly_binding"]
    assert r["release_dnll_joint"] == np.inf


def test_upper_bound_and_non_binding_rows():
    """The same face written as an upper bound (-c <= 0) binds with mu < 0; a
    row near its bound whose multiplier has the non-binding sign is dropped."""
    g, H, cval, J, chess = _problem()
    r = face_held_covariance(
        g, H, -cval, -J, [-np.inf], [0.0], ["-c"], lambda v: -chess(v)
    )
    (face,) = r["active"]
    assert face["multiplier"] == pytest.approx(-1.0)
    # a second row y >= 0 at its bound, with the loss pulling y INTO the
    # allowed region (multiplier of the wrong sign): not binding, dropped
    J2 = np.vstack([J, [0.0, 1.0]])
    r2 = face_held_covariance(
        np.array([g[0], -0.5]),
        H,
        np.array([0.0, 0.0]),
        J2,
        [0.0, 0.0],
        [np.inf, np.inf],
        ["c", "y"],
        lambda v: v[0] * np.diag([0.0, -1.4]),
    )
    assert [a["label"] for a in r2["active"]] == ["c"]
    # with the loss flat across it (zero multiplier) the row is kept, but it is
    # flagged weakly binding with a zero release gain: the interval along it is
    # one-sided
    r3 = face_held_covariance(
        g,
        H,
        np.array([0.0, 0.0]),
        J2,
        [0.0, 0.0],
        [np.inf, np.inf],
        ["c", "y"],
        lambda v: v[0] * np.diag([0.0, -1.4]),
    )
    (y_face,) = [a for a in r3["active"] if a["label"] == "y"]
    assert y_face["release_dnll"] == pytest.approx(0.0, abs=1e-15)
    assert y_face["weakly_binding"]


def test_two_faces_release_one_with_the_other_held():
    """Two independent linear faces on a diagonal quadratic: releasing either
    alone gains exactly its own 1/2 g_i^2 / h_i."""
    h = np.array([2.0, 3.0, 1.0])
    g = np.array([-1.0, -2.0, 0.0])  # pushing x0 and x1 against x_i <= 0
    H = np.diag(h)
    J = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    r = face_held_covariance(
        g,
        H,
        np.zeros(2),
        J,
        [0.0, 0.0],
        [np.inf, np.inf],
        ["x0", "x1"],
        lambda v: np.zeros((3, 3)),
    )
    gains = [a["release_dnll"] for a in r["active"]]
    np.testing.assert_allclose(gains, 0.5 * g[:2] ** 2 / h[:2])
    assert r["release_dnll_joint"] == pytest.approx(sum(gains))
    np.testing.assert_allclose(r["cov_held"], np.diag([0.0, 0.0, 1.0]), atol=1e-14)


def test_frozen_block_comes_from_cov_base():
    g, H, cval, J, chess = _problem()
    g3 = np.append(g, 0.0)
    H3 = np.diag([1.0, 1.0, 5.0])
    J3 = np.append(J, [[0.0]], axis=1)
    base = np.diag([9.0, 9.0, 7.0])
    r = face_held_covariance(
        g3,
        H3,
        cval,
        J3,
        [0.0],
        [np.inf],
        ["c"],
        lambda v: np.pad(chess(v), ((0, 1), (0, 1))),
        floating=[0, 1],
        cov_base=base,
    )
    assert r["cov_held"][2, 2] == 7.0
    assert r["cov_held"][0, 2] == 0.0  # base off-diagonal
