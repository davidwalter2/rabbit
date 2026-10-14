"""Hard constraints for the constrained minimizers (``--minimizerMethod trust-constr``).

A :class:`Constraints` object declares a set of conditions

    lower <= c(x) <= upper

on the fit's parameters, which the minimizer then honours EXACTLY, instead of
the "fake" enforcement of a penalty that only grows outside the region. It is
a template class, like :class:`rabbit.param_models.param_model.ParamModel`:
users write their own subclass and load it with
``--constraints module.Class [args...]``.

Why its own class, and not part of the ParamModel or of a regularizer:

* a constraint can act on ANY fit parameter -- a POI, a model parameter, or a
  nuisance from the card -- so it cannot belong to the param model;
* a regularizer (``-r``) adds a term to -ln(L) and must stay in the loss under
  every minimizer (a lattice chi^2 is data, not a region), so constraints must
  not be smuggled through it either.

What a subclass provides (see :class:`Bounds` for a complete example):

* :meth:`set_layout` -- resolve parameter NAMES to positions in the full
  ``[ParamModel params | systematics]`` vector. Called on every layout change
  (fit initialisation, the saturated test's re-initialisation); never cache
  positions across calls.
* :attr:`labels`, :attr:`lower`, :attr:`upper` -- one entry per condition row.
  ``-inf`` / ``+inf`` mark a one-sided row; ``lower == upper`` an equality.
* :meth:`values` -- ``c(x)`` as a 1-D TensorFlow tensor. ``x`` is the PHYSICAL
  vector the likelihood sees (``Fitter.get_x()``: blinding offsets and the POI
  positivity transform applied). It must be twice differentiable: rabbit takes
  its Jacobian and its Hessian-vector products by autodiff, and hands both to
  the minimizer.
* :meth:`parameter_indices` (optional) -- the positions of the parameters
  :meth:`values` reads, so the post-fit can restrict the constraint Hessian.

Everything optimizer-specific (scipy's ``NonlinearConstraint``, which rows are
held feasible, frozen parameters) is the fitter's job, not the subclass's.
"""

import re

import numpy as np
import tensorflow as tf


class Constraints:
    """Base class; see the module docstring for the contract."""

    def __init__(self, indata, *args, **kwargs):
        self.indata = indata
        self.dtype = indata.dtype
        self.labels = []
        self.lower = np.zeros(0)
        self.upper = np.zeros(0)

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        """Build from the ``--constraints`` command-line tokens."""
        return cls(indata, *args, **kwargs)

    def set_layout(self, parms):
        """Resolve positions in the fit vector from its parameter names ``parms``."""
        raise NotImplementedError

    def values(self, x):
        """``c(x)``: a 1-D tensor, one entry per row of :attr:`labels`."""
        raise NotImplementedError

    def parameter_indices(self):
        """Positions (in the current layout) of every parameter ``values`` reads.

        Optional but recommended: the post-fit uses it to restrict the dense
        constraint Hessian to these columns. ``None`` (the default) means
        "unknown", and rabbit then differentiates with respect to every
        floating parameter -- always correct, but one backward pass per
        parameter.
        """
        return None

    @staticmethod
    def resolve_indices(parms, names, who="Constraints"):
        """Map parameter names to positions in ``parms``, raising if any is absent."""
        parms = np.asarray(parms).astype(str)
        index = {name: i for i, name in enumerate(parms)}
        missing = [n for n in names if n not in index]
        if missing:
            raise ValueError(f"{who}: {missing} not in the fit's parameter vector")
        return {n: index[n] for n in names}


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf"
_BOUND_RE = [
    # a<=name<=b
    (re.compile(rf"^\s*({_NUM})\s*<=\s*([^<>=\s]+)\s*<=\s*({_NUM})\s*$"), "both"),
    # name>=v
    (re.compile(rf"^\s*([^<>=\s]+)\s*>=\s*({_NUM})\s*$"), "lower"),
    # name<=v
    (re.compile(rf"^\s*([^<>=\s]+)\s*<=\s*({_NUM})\s*$"), "upper"),
]


class Bounds(Constraints):
    """Simple bounds on named parameters, e.g.

        --constraints Bounds 'lambda2_nu>=0' 'pdfEig3<=2' '-1<=bkg_norm<=1'

    in the parameter's own fit units (what ``rabbit_fit`` reports for it).
    Quote the tokens: ``<`` and ``>`` are shell redirections.
    """

    def __init__(self, indata, *args, **kwargs):
        super().__init__(indata)
        if not args:
            raise ValueError(
                "Bounds: give at least one 'name>=v', 'name<=v' or 'a<=name<=b'"
            )
        names, lows, ups = [], [], []
        for tok in args:
            for regex, kind in _BOUND_RE:
                m = regex.match(tok)
                if m is None:
                    continue
                if kind == "both":
                    lo, name, up = float(m.group(1)), m.group(2), float(m.group(3))
                elif kind == "lower":
                    name, lo, up = m.group(1), float(m.group(2)), np.inf
                else:
                    name, lo, up = m.group(1), -np.inf, float(m.group(2))
                break
            else:
                raise ValueError(
                    f"Bounds: cannot parse '{tok}'; expected 'name>=v', "
                    "'name<=v' or 'a<=name<=b'"
                )
            if not lo <= up:
                raise ValueError(f"Bounds: empty interval in '{tok}'")
            names.append(name)
            lows.append(lo)
            ups.append(up)
        if len(set(names)) != len(names):
            raise ValueError(f"Bounds: a parameter appears twice in {list(args)}")
        self.names = names
        self.labels = [tok.strip() for tok in args]
        self.lower = np.asarray(lows, dtype=np.float64)
        self.upper = np.asarray(ups, dtype=np.float64)
        self._idx = None

    def set_layout(self, parms):
        index = self.resolve_indices(parms, self.names, "Bounds")
        self._pos = [index[n] for n in self.names]
        self._idx = tf.constant(self._pos, dtype=tf.int32)

    def values(self, x):
        return tf.gather(x, self._idx)

    def parameter_indices(self):
        return list(self._pos)
