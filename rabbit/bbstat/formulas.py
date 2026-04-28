"""Closed-form helpers for the bin-by-bin (BBB) statistical treatment.

Pure functions only — no TF Variables, no Fitter state. The algebra here
is shared between :mod:`rabbit.bbstat.bbstat` and the Newton-loop
helpers in :mod:`rabbit.bbstat.newton`.
"""

import tensorflow as tf


def solve_quad_eq(a, b, c):
    """Positive root of ``a x² + b x + c = 0`` (TF-broadcast)."""
    return 0.5 * (-b + tf.sqrt(b**2 - 4.0 * a * c)) / a
