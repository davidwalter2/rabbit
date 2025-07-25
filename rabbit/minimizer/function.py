import tensorflow as tf
import numpy as np
from collections import namedtuple
import time

sf_value = namedtuple("sf_value", ["f", "grad", "hessp", "hess"])
de_value = namedtuple("de_value", ["f", "grad"])


class ScalarFunction:
    """Scalar-valued objective function with autodiff backend (TensorFlow)."""

    def __new__(cls, fun, hessp=False, hess=False):# twice_diffable=True):
        if isinstance(fun, ScalarFunction):
            assert fun._hessp == hessp
            assert fun._hess == hess
            return fun
        return super().__new__(cls)

    def __init__(self, fun, hessp=False, hess=False):#, twice_diffable=True):
        self.fun = fun
        # self._x_shape = x_shape
        self._hessp = hessp
        self._hess = hess

        self.time_closure = 0
        self.n_closure = 0
        # self._twice_diffable = twice_diffable
        # self.nfev = 0
        # self._I = None

    # def fun(self, x):
    #     x = tf.reshape(x, self._x_shape)
    #     f = self._fun(x)
    #     if tf.size(f) != 1:
    #         raise RuntimeError("ScalarFunction was supplied a function that does not return scalar outputs.")
    #     self.nfev += 1
    #     return f
    def closure(self, x):
        start = time.time()
        results = self._closure(x)

        self.time_closure += time.time() - start

        self.n_closure += 1

        return results
    
    @tf.function
    def _closure(self, x):
        # x = tf.Variable(tf.reshape(x, self._x_shape))

        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                val = self.fun(x)
            grad = tape1.gradient(val, x)

        if self._hess:
            hess = tape2.jacobian(grad, x)
        else:
            hess = None

        # with tf.GradientTape() as t2:
        #     with tf.GradientTape() as t1:
        #         val = self._compute_loss(profile=profile)
        #     grad = t1.gradient(val, self.x)
        # hess = t2.jacobian(grad, self.x)
        # if (self._hess or self._hessp) and grad is None:
        #     raise RuntimeError("A 2nd-order derivative was requested but the objective is not twice-differentiable.")

        if self._hessp:
            def hessp_func(v):
                return tape2.gradient(tf.tensordot(grad, v, axes=1), x)
            hessp = hessp_func  # returns a callable
        else:
            hessp = None

        # if self._hess:
        #     if self._I is None:
        #         n = tf.size(x)
        #         self._I = tf.eye(n, dtype=x.dtype)
        #     def single_hvp(v):
        #         return tape2.gradient(tf.tensordot(grad, v, axes=1), x)
        #     hess = tf.stack([single_hvp(self._I[i]) for i in range(self._I.shape[0])])

        return val, grad, hessp, hess


        # return sf_value(f=val.numpy(), grad=grad.numpy(), hessp=hessp, hess=hess.numpy() if hess is not None else None)

    # def dir_evaluate(self, x, t, d):
    #     """Evaluate f(x + t*d) and its gradient."""
    #     x = tf.Variable(x + t * d)

    #     with tf.GradientTape() as tape:
    #         f = self.fun(x)
    #     grad = tape.gradient(f, x)

    #     return de_value(f=float(f), grad=grad)